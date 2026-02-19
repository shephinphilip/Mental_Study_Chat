#!/usr/bin/env python3
"""
Dr. Mind v6.0 — main.py
Full FastAPI application with Production Hardening

Production Strategies Integrated:
  1.  Token Bucket Rate Limiting (TPM + RPM)     → via LLMManager
  2.  Concurrency Control via Async Semaphore     → chat endpoint
  3.  Priority-Based Request Handling             → chat endpoint
  4.  Singleton LLM Manager                       → crisis.py LLMManager
  5.  Automatic Fallback Model Switching          → crisis.py LLMManager
  6.  Parallel Preprocessing (LangGraph)          → crisis.py preprocess_node
  7.  Router Model Isolation                      → crisis.py separate instances
  8.  Global Thread Pool                          → crisis.py _thread_pool
  9.  Ring-Buffer Metrics                         → crisis.py MetricsRing
  10. Endpoint-Level Hard Timeout Guard           → chat endpoint
  11. Eager Startup Initialization                → lifespan()
  12. Health Endpoint with Readiness Logic        → /api/health
  13. Regex Hard Gates Before LLM                 → crisis.py _rule_prefilter
  14. Bounded Conversation History                → chat endpoint + crisis.py
  15. Production Launch Configuration             → Config class

Endpoints:
  ── Session ──────────────────────────────────────────────────────
  POST /api/session/start            Start a new clinical session
  POST /api/session/clear            Clear in-memory history (keep session alive)
  GET  /api/profiles                 List available student profiles

  ── Chat ─────────────────────────────────────────────────────────
  POST /api/chat                     Send a message, receive clinical response

  ── Conversation History ──────────────────────────────────────────
  POST /api/conversations/save       Explicitly persist a full session to MongoDB
  GET  /api/conversations/{user_id}  All saved sessions for a user
  GET  /api/conversations/{user_id}/messages  Flat per-message history
  DELETE /api/conversations/{session_id}      Delete a session

  ── Meditations ──────────────────────────────────────────────────
  GET  /api/meditations              List all meditations
  GET  /api/meditations/suggest/{classification}  Suggest for a classification
  POST /api/meditations/progress/start  Log session start
  POST /api/meditations/progress/complete  Mark session complete
  GET  /api/meditations/streaks/{user_id}  Get streak data
  GET  /api/meditations/audios       Fetch active audio tracks

  ── Router Memory ────────────────────────────────────────────────
  GET  /api/memory/{student_id}      Get student's router memory

  ── Production Observability ─────────────────────────────────────
  GET  /api/health                   Readiness + provider health + metrics
  GET  /api/metrics                  Detailed ring-buffer metrics
  GET  /api/status                   Quick liveness probe

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import LLMResult
from pydantic import BaseModel

# ── Import production crisis engine ──────────────────────────────────────────
from crisis import (
    build_agent,
    MEDITATION_MAP,
    MEDITATIONS,
    Config,
    LLMManager,
    MetricsRing,
    RequestMetric,
    Priority,
    Priority,
    _rule_prefilter,
    _thread_pool,
)
from db import db

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dr_mind")


# ── Pricing (updated for Gemini primary + OpenAI fallback) ────────────────────
# Gemini 2.5 Flash:  \$0.15 / 1M input,  \$0.60 / 1M output (approx)
# gpt-4o-mini:       \$0.15 / 1M input,  \$0.60 / 1M output
# gpt-4o:            \$5.00 / 1M input,  \$15.00 / 1M output
GEMINI_IN  = 0.15  / 1_000_000
GEMINI_OUT = 0.60  / 1_000_000
MINI_IN    = 0.15  / 1_000_000
MINI_OUT   = 0.60  / 1_000_000
GPT4O_IN   = 5.00  / 1_000_000
GPT4O_OUT  = 15.00 / 1_000_000


# ── Token tracker ─────────────────────────────────────────────────────────────

class TokenTracker(BaseCallbackHandler):
    """Accumulates token usage across all LLM calls in a single agent.invoke()."""

    def __init__(self):
        self.llm_calls:         int  = 0
        self.prompt_tokens:     int  = 0
        self.completion_tokens: int  = 0
        self._call_log:         list = []

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.llm_calls += 1
        usage = (response.llm_output or {}).get("token_usage", {})
        pt = usage.get("prompt_tokens",     0)
        ct = usage.get("completion_tokens", 0)
        self.prompt_tokens     += pt
        self.completion_tokens += ct
        self._call_log.append({"call": self.llm_calls, "prompt": pt, "completion": ct})

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        if self.llm_calls == 0:
            return 0.0
        # Classifier call (gemini/mini) — assume 20 completion tokens
        mini_est_pt = min(self.prompt_tokens, 900)
        mini_est_ct = 20
        remaining_pt = max(0, self.prompt_tokens - mini_est_pt)
        remaining_ct = max(0, self.completion_tokens - mini_est_ct)
        return (
            mini_est_pt  * GEMINI_IN  + mini_est_ct  * GEMINI_OUT +
            remaining_pt * GEMINI_IN  + remaining_ct * GEMINI_OUT
        )

    def summary(self) -> dict:
        return {
            "llm_calls":         self.llm_calls,
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.total_tokens,
            "cost_usd":          round(self.cost_usd, 6),
            "per_call_detail":   self._call_log,
        }


# ── In-memory session store ───────────────────────────────────────────────────
# {session_id: {agent, history, profile, detected_language, created, ...}}
SESSIONS: Dict[str, dict] = {}


# ═════════════════════════════════════════════════════════════════
# PRODUCTION GLOBALS — initialized at startup (Strategy 11)
# ═════════════════════════════════════════════════════════════════
_llm_manager: LLMManager | None = None
_metrics: MetricsRing | None = None
_concurrency_semaphore: asyncio.Semaphore | None = None
_startup_time: float = 0.0


# ═════════════════════════════════════════════════════════════════
# STRATEGY 11: EAGER STARTUP INITIALIZATION (lifespan)
# ═════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Strategy 11: Initialize ALL production systems at startup.
    - LLM connections (Strategy 4: Singleton)
    - Rate limiters (Strategy 1: Token Bucket)
    - Metrics ring buffer (Strategy 9)
    - Priority queue (Strategy 3)
    - Concurrency semaphore (Strategy 2)
    - Thread pool warm-up (Strategy 8)
    - MongoDB connectivity check
    """
    global _llm_manager, _metrics, _concurrency_semaphore, _startup_time

    log.info("=" * 65)
    log.info("  Dr. Mind v6.0 — Production Backend Starting")
    log.info("=" * 65)

    _startup_time = time.time()

    # ── Strategy 4: Singleton LLM Manager ──────────────────────
    _llm_manager = LLMManager()
    _llm_manager.initialize()

    # ── Strategy 9: Ring-Buffer Metrics ────────────────────────
    _metrics = MetricsRing(maxlen=Config.METRICS_BUFFER_SIZE)

    # ── Strategy 2: Concurrency Semaphore ──────────────────────
    _concurrency_semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT)

    # ── Strategy 8: Warm up thread pool ────────────────────────
    _thread_pool.submit(lambda: None).result()

    # ── Validate API keys ──────────────────────────────────────
    has_gemini = bool(Config.GOOGLE_API_KEY)
    has_openai = bool(Config.OPENAI_API_KEY)

    if not has_gemini and not has_openai:
        log.error("❌ No LLM API keys configured! Set GOOGLE_API_KEY and/or OPENAI_API_KEY")
        raise RuntimeError("No LLM API keys configured")

    if not has_gemini:
        log.warning("⚠️  GOOGLE_API_KEY not set — running on OpenAI only (higher cost)")
    if not has_openai:
        log.warning("⚠️  OPENAI_API_KEY not set — no fallback available")

    # ── MongoDB check ──────────────────────────────────────────
    mongo_ok = await db.ping()
    if not mongo_ok:
        log.warning("⚠️  MongoDB not reachable — check MONGODB_URI in .env")
    else:
        log.info("✅ MongoDB (zenark) connected")

    # ── Log configuration ──────────────────────────────────────
    log.info(f"  Primary model:   {Config.GEMINI_MODEL}")
    log.info(f"  Fallback models: {Config.OPENAI_CLASSIFIER} / {Config.OPENAI_RESPONDER}")
    log.info(f"  Max concurrent:  {Config.MAX_CONCURRENT}")
    log.info(f"  Thread pool:     {Config.THREAD_POOL_SIZE}")
    log.info(f"  Request timeout: {Config.REQUEST_TIMEOUT_SEC}s")
    log.info(f"  History cap:     {Config.MAX_HISTORY_MESSAGES} msgs")
    log.info(f"  Metrics buffer:  {Config.METRICS_BUFFER_SIZE}")
    log.info(f"  Gemini RPM/TPM:  {Config.GEMINI_RPM}/{Config.GEMINI_TPM}")
    log.info(f"  OpenAI RPM/TPM:  {Config.OPENAI_RPM}/{Config.OPENAI_TPM}")
    log.info("=" * 65)
    log.info("  ✅ Dr. Mind v6.0 Production API Ready")
    log.info("=" * 65)

    yield

    # ── Shutdown ───────────────────────────────────────────────
    log.info("Shutting down thread pool...")
    _thread_pool.shutdown(wait=True, cancel_futures=False)
    log.info("Goodbye.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Dr. Mind Clinical API",
    description="Production psychiatric interview engine for Indian teenagers — zenark backend",
    version="6.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    student_id: str
    user_id:    Optional[str] = None


class StartSessionResponse(BaseModel):
    session_id:    str
    student_id:    str
    name:          str
    risk_level:    str
    age:           int
    chief_concern: str


class ChatRequest(BaseModel):
    session_id: str
    message:    str
    user_id:    Optional[str] = None


class MeditationSuggestion(BaseModel):
    id:          int
    title:       str
    description: str
    relevance:   str


class UsageInfo(BaseModel):
    llm_calls:         int
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int
    cost_usd:          float
    per_call_detail:   List[dict]


class ChatResponse(BaseModel):
    response:              str
    classification:        str
    inquiry_stage:         str
    session_id:            str
    detected_language:     str
    usage:                 UsageInfo
    meditation_suggestion: Optional[MeditationSuggestion] = None
    latency_ms:            float = 0.0
    provider:              Optional[str] = None


class ClearRequest(BaseModel):
    session_id: str


class SaveConversationRequest(BaseModel):
    session_id: str
    user_id:    str


class MeditationProgressStartRequest(BaseModel):
    user_id:       str
    meditation_id: int


class MeditationProgressCompleteRequest(BaseModel):
    user_id:       str
    meditation_id: int
    time_spent:    int


# ── Helper functions ──────────────────────────────────────────────────────────

def _msgs_to_dicts(messages) -> List[Dict]:
    out = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append({"role": "student",  "content": m.content})
        elif isinstance(m, AIMessage):
            out.append({"role": "dr_mind",  "content": m.content})
        else:
            out.append({"role": "system",   "content": str(m.content)})
    return out


async def _persist_session(session_id: str, user_id: str, session: dict) -> None:
    msgs_raw = _msgs_to_dicts(session["history"])
    await db.chats.upsert_session(
        session_id=session_id,
        user_id=user_id,
        messages=msgs_raw,
    )
    log.info("💾 session %s persisted to MongoDB chats (%d msgs)", session_id, len(msgs_raw))


async def _append_normalized(
    student_id:     str,
    speaker:        str,
    text:           str,
    classification: str,
) -> None:
    await db.normalized.append_message(
        student_id=student_id,
        speaker=speaker,
        message_text=text,
        labels=[classification],
    )


def _get_meditation_suggestion(classification: str) -> Optional[MeditationSuggestion]:
    entry = MEDITATION_MAP.get(classification)
    if not entry:
        return None
    primary_id = entry["ids"][0]
    med = MEDITATIONS.get(primary_id)
    if not med:
        return None
    return MeditationSuggestion(
        id=primary_id,
        title=med["title"],
        description=med["description"],
        relevance=entry["relevance"],
    )


# ── MOCK_STUDENTS import ──────────────────────────────────────────────────────
try:
    from models import MOCK_STUDENTS
except ImportError:
    MOCK_STUDENTS = {}
    log.warning("⚠️  models.py not found — /api/profiles will return empty list")


# ═════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═════════════════════════════════════════════════════════════════


# ── Strategy 12: Health Endpoint with Readiness Logic ─────────────────────────

@app.get("/api/health")
async def health():
    """
    Strategy 12: Production health check with readiness logic.
    Used by load balancers, K8s probes, and monitoring.
    """
    """
    Strategy 12: Production health check with readiness logic.
    Used by load balancers, K8s probes, and monitoring.
    """

    mongo_ok = await db.ping()

    if _llm_manager is None or _metrics is None:
        return {
            "status":          "unavailable",
            "version":         "6.0.0",
            "mongodb":         "connected" if mongo_ok else "unreachable",
            "sessions_active": len(SESSIONS),
            "uptime_sec":      0.0,
        }

    provider_health = _llm_manager.health_status
    has_any = provider_health.get("has_gemini_key") or provider_health.get("has_openai_key")

    if not has_any:
        status = "unavailable"
    elif not provider_health.get("gemini_healthy", True):
        status = "degraded"
    elif not mongo_ok:
        status = "degraded"
    else:
        status = "ready"

    metrics_summary = _metrics.summary(window_sec=60)

    return {
        "status":          status,
        "version":         "6.0.0",
        "mongodb":         "connected" if mongo_ok else "unreachable",
        "sessions_active": len(SESSIONS),
        "uptime_sec":      round(time.time() - _startup_time, 1),
        "providers":       provider_health,
        "concurrency": {
            "active": Config.MAX_CONCURRENT - _concurrency_semaphore._value,
            "max":    Config.MAX_CONCURRENT,
        },
        "metrics_60s":     metrics_summary.get(f"last_60s", {}),
    }


@app.get("/api/metrics")
async def metrics_endpoint():
    """Strategy 9: Detailed ring-buffer metrics for monitoring dashboards."""
    if _metrics is None:
        raise HTTPException(status_code=503, detail="Not initialized")
    return {
        "summary_60s":  _metrics.summary(60),
        "summary_300s": _metrics.summary(300),
        "buffer_usage": _metrics.buffer_usage,
        "providers":    _llm_manager.health_status if _llm_manager else {},
    }


@app.get("/api/status")
async def quick_status():
    """Lightweight liveness probe — no DB call."""
    return {"alive": True, "version": "6.0.0", "uptime_sec": round(time.time() - _startup_time, 1)}


# ── Profiles ──────────────────────────────────────────────────────────────────

@app.get("/api/profiles")
async def list_profiles():
    result = []
    for sid, profile in MOCK_STUDENTS.items():
        result.append({
            "student_id":    sid,
            "name":          profile.name,
            "age":           profile.age,
            "current_class": profile.current_class,
            "city":          profile.city,
            "risk_level":    profile.current_risk_level.value,
            "chief_concern": profile.chief_concern,
        })
    return result


# ── Session ───────────────────────────────────────────────────────────────────

@app.post("/api/session/start", response_model=StartSessionResponse)
async def start_session(req: StartSessionRequest):
    profile = MOCK_STUDENTS.get(req.student_id)
    if not profile:
        raise HTTPException(404, f"Student '{req.student_id}' not found")

    session_id = hashlib.md5(
        f"{req.student_id}_{datetime.now(timezone.utc).isoformat()}".encode()
    ).hexdigest()[:16]

    agent = build_agent(profile)

    SESSIONS[session_id] = {
        "agent":             agent,
        "history":           [],
        "profile":           profile,
        "user_id":           req.user_id or req.student_id,
        "detected_language": "ENGLISH",
        "classifications":   [],
        "created":           datetime.now(timezone.utc),
    }

    log.info("🚀 session %s started for student %s", session_id, req.student_id)
    return StartSessionResponse(
        session_id=session_id,
        student_id=profile.student_id,
        name=profile.name,
        risk_level=profile.current_risk_level.value,
        age=profile.age,
        chief_concern=profile.chief_concern,
    )


@app.post("/api/session/clear")
async def clear_session(req: ClearRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found.")
    session["history"]         = []
    session["classifications"] = []
    return {"status": "cleared", "session_id": req.session_id}


# ═════════════════════════════════════════════════════════════════
# CHAT ENDPOINT — All production strategies converge here
#
# Strategy 2:  Concurrency semaphore
# Strategy 3:  Priority-based handling
# Strategy 10: Hard timeout guard
# Strategy 14: Bounded history
# ═════════════════════════════════════════════════════════════════

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):

    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found. Call /api/session/start first.")

    message = req.message.strip()
    if not message:
        raise HTTPException(400, "Message cannot be empty.")

    t_start = time.time()

    agent   = session["agent"]
    history = session["history"]
    profile = session["profile"]
    user_id = req.user_id or session.get("user_id", profile.student_id)

    # ── Strategy 14: Bound history before appending ────────────
    if len(history) >= Config.MAX_HISTORY_MESSAGES:
        history = history[-(Config.MAX_HISTORY_MESSAGES - 1):]
        session["history"] = history

    history.append(HumanMessage(content=message))

    # ── Strategy 3: Quick priority classification via regex ────
    regex_cls = _rule_prefilter(message)
    priority = Priority.from_classification(regex_cls or "SAFE")

    # ── Strategy 2: Concurrency control ────────────────────────
    # Safety/crisis requests always proceed; others can be rejected at capacity
    if _concurrency_semaphore._value <= 0 and priority > Priority.SAFETY:
        if _metrics:
            _metrics.increment("rejected_at_capacity")
        raise HTTPException(
            429,
            "Server at capacity. Please try again shortly."
        )

    prev_language = session.get("detected_language", "ENGLISH")
    tracker       = TokenTracker()

    async with _concurrency_semaphore:
        # ── Strategy 10: Hard timeout guard ────────────────────
        try:
            result = await asyncio.wait_for(
                _run_agent_async(agent, history, message, prev_language, profile, tracker),
                timeout=Config.REQUEST_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            if _metrics:
                _metrics.increment("timeouts")
            latency = (time.time() - t_start) * 1000
            log.error(
                "⏱️  TIMEOUT after %.0fms | student=%s msg='%s'",
                latency, profile.student_id, message[:50],
            )
            raise HTTPException(
                504,
                f"Request timed out after {Config.REQUEST_TIMEOUT_SEC}s. Please try again."
            )

    # ── Extract results ────────────────────────────────────────
    response_text     = result["messages"][-1].content
    classification    = result.get("classification", "SAFE")
    inquiry_stage     = result.get("inquiry_stage", "initial")
    detected_language = result.get("detected_language", prev_language)
    usage             = tracker.summary()
    latency_ms        = (time.time() - t_start) * 1000

    # ── Strategy 9: Record metrics ─────────────────────────────
    if _metrics:
        _metrics.record(RequestMetric(
            timestamp=t_start,
            classification=classification,
            provider="tracked",
            node="chat_endpoint",
            latency_ms=latency_ms,
            tokens_estimated=usage["total_tokens"],
            success=True,
        ))

    # ── Log usage ──────────────────────────────────────────────
    log.info(
        "💬 student=%-10s  cls=%-20s  lang=%-7s  %.0fms  $%.6f",
        profile.student_id, classification, detected_language,
        latency_ms, usage["cost_usd"],
    )
    if usage["llm_calls"] > 0:
        log.info(
            "📊 llm_calls=%d  prompt=%d  completion=%d  total=%d",
            usage["llm_calls"], usage["prompt_tokens"],
            usage["completion_tokens"], usage["total_tokens"],
        )
        for c in usage["per_call_detail"]:
            log.debug("   ↳ call #%d  prompt=%d  completion=%d", c["call"], c["prompt"], c["completion"])

    # ── Meditation suggestion ──────────────────────────────────
    suggestion = _get_meditation_suggestion(classification)

    # ── Update session state ───────────────────────────────────
    session["detected_language"] = detected_language
    history.append(AIMessage(content=response_text))

    # Strategy 14: Enforce rolling window
    session["history"] = history[-Config.MAX_HISTORY_MESSAGES:]
    session["classifications"].append(classification)

    # ── MongoDB: append normalized messages ────────────────────
    try:
        await _append_normalized(profile.student_id, "student",  message,       classification)
        await _append_normalized(profile.student_id, "dr_mind",  response_text, classification)
    except Exception as e:
        log.warning("⚠️  normalized append failed: %s", e)

    # ── MongoDB: upsert router memory ──────────────────────────
    try:
        await db.router_memory.upsert(
            student_id=profile.student_id,
            session_id=req.session_id,
            last_tool=classification,
            last_emotion=classification,
            preferred_language=detected_language,
            dominant_emotions=list(set(session["classifications"][-10:])),
            recurring_topics=list(set(session["classifications"])),
        )
    except Exception as e:
        log.warning("⚠️  router_memory upsert failed: %s", e)

    # ── MongoDB: auto-persist full chat every 5 turns ──────────
    if len(history) % 10 == 0:
        try:
            await _persist_session(req.session_id, user_id, session)
        except Exception as e:
            log.warning("⚠️  auto-persist failed: %s", e)

    return ChatResponse(
        response=response_text,
        classification=classification,
        inquiry_stage=inquiry_stage,
        session_id=req.session_id,
        detected_language=detected_language,
        usage=UsageInfo(**usage),
        meditation_suggestion=suggestion,
        latency_ms=round(latency_ms, 1),
    )


async def _run_agent_async(
    agent,
    history: list,
    message: str,
    prev_language: str,
    profile,
    tracker: TokenTracker,
) -> dict:
    """
    Execute the LangGraph agent.
    Strategy 8: Runs in global thread pool (LangGraph invoke is sync).
    """
    invoke_input = {
        "messages":          history,
        "classification":    "SAFE",
        "inquiry_stage":     "initial",
        "current_input":     message,
        "crisis_verified":   False,
        "detected_language": prev_language,
    }

    # LangGraph .invoke() is synchronous — run in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _thread_pool,
        lambda: agent.invoke(invoke_input, config={"callbacks": [tracker]}),
    )
    return result


# ── Conversation History ──────────────────────────────────────────────────────

@app.post("/api/conversations/save")
async def save_conversation(req: SaveConversationRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found.")
    try:
        await _persist_session(req.session_id, req.user_id, session)
    except Exception as e:
        log.error("❌ persist failed: %s", e)
        raise HTTPException(500, f"MongoDB persist failed: {e}")
    return {
        "status":     "saved",
        "session_id": req.session_id,
        "messages":   len(session["history"]),
    }


@app.get("/api/conversations/{user_id}")
async def get_conversations(
    user_id: str,
    limit:   int = Query(default=20, ge=1, le=100),
):
    try:
        sessions = await db.chats.get_user_sessions(user_id, limit=limit)
    except Exception as e:
        raise HTTPException(500, f"MongoDB error: {e}")
    result = []
    for s in sessions:
        msgs = s.get("messages", [])
        result.append({
            "session_id":    s.get("session_id"),
            "timestamp":     str(s.get("timestamp", "")),
            "message_count": len(msgs),
            "preview":       msgs[0]["content"][:80] if msgs else "",
        })
    return result


@app.get("/api/conversations/{user_id}/messages")
async def get_conversation_messages(
    user_id:    str,
    session_id: Optional[str] = Query(default=None),
    limit:      int = Query(default=50, ge=1, le=200),
):
    try:
        messages = await db.normalized.get_student_history(user_id, limit=limit)
    except Exception as e:
        raise HTTPException(500, f"MongoDB error: {e}")
    return {"student_id": user_id, "messages": messages, "count": len(messages)}


@app.get("/api/conversations/session/{session_id}")
async def get_session_detail(session_id: str):
    try:
        session = await db.chats.get_session(session_id)
    except Exception as e:
        raise HTTPException(500, f"MongoDB error: {e}")
    if not session:
        raise HTTPException(404, "Session not found in MongoDB.")
    return session


@app.delete("/api/conversations/{session_id}")
async def delete_conversation(session_id: str):
    try:
        deleted = await db.chats.delete_session(session_id)
    except Exception as e:
        raise HTTPException(500, f"MongoDB error: {e}")
    if not deleted:
        raise HTTPException(404, "Session not found.")
    SESSIONS.pop(session_id, None)
    return {"status": "deleted", "session_id": session_id}


# ── Meditations ───────────────────────────────────────────────────────────────

@app.get("/api/meditations")
async def list_meditations():
    return {"meditations": list(MEDITATIONS.values()), "count": len(MEDITATIONS)}


@app.get("/api/meditations/suggest/{classification}")
async def suggest_meditation(classification: str):
    suggestion = _get_meditation_suggestion(classification.upper())
    if not suggestion:
        raise HTTPException(404, f"No meditation mapping for '{classification}'")
    return suggestion


@app.post("/api/meditations/progress/start")
async def meditation_start(req: MeditationProgressStartRequest):
    med = MEDITATIONS.get(req.meditation_id)
    if not med:
        raise HTTPException(404, f"Meditation id {req.meditation_id} not found")
    try:
        session_id = f"med_{req.user_id}_{req.meditation_id}_{int(datetime.now().timestamp())}"
        doc_id = await db.meditation_progress.start(req.user_id, session_id)
    except Exception as e:
        raise HTTPException(500, f"MongoDB error: {e}")
    return {"status": "started", "progress_id": doc_id, "meditation": med["title"]}


@app.post("/api/meditations/progress/complete")
async def meditation_complete(req: MeditationProgressCompleteRequest):
    med = MEDITATIONS.get(req.meditation_id)
    if not med:
        raise HTTPException(404, f"Meditation id {req.meditation_id} not found")
    session_id = f"med_{req.user_id}_{req.meditation_id}"
    try:
        await db.meditation_progress.complete(req.user_id, session_id, req.time_spent)
        streak = await db.meditation_streaks.update_after_session(req.user_id)
    except Exception as e:
        raise HTTPException(500, f"MongoDB error: {e}")
    return {
        "status":       "completed",
        "meditation":   med["title"],
        "time_spent_s": req.time_spent,
        "streak": {
            "current": streak.get("current_streak"),
            "longest": streak.get("longest_streak"),
        },
    }


@app.get("/api/meditations/history/{user_id}")
async def meditation_history(
    user_id: str,
    limit:   int = Query(default=20, ge=1, le=100),
):
    try:
        history = await db.meditation_progress.get_user_history(user_id, limit=limit)
    except Exception as e:
        raise HTTPException(500, f"MongoDB error: {e}")
    return {"user_id": user_id, "history": history}


@app.get("/api/meditations/streaks/{user_id}")
async def meditation_streaks(user_id: str):
    try:
        streak = await db.meditation_streaks.get(user_id)
    except Exception as e:
        raise HTTPException(500, f"MongoDB error: {e}")
    return streak or {"user_id": user_id, "current_streak": 0, "longest_streak": 0}


@app.get("/api/meditations/audios")
async def meditation_audios(genre: Optional[str] = Query(default=None)):
    try:
        if genre:
            tracks = await db.audios.get_by_genre(genre)
        else:
            tracks = await db.audios.get_active()
    except Exception as e:
        raise HTTPException(500, f"MongoDB error: {e}")
    return {"tracks": tracks, "count": len(tracks)}


# ── Router Memory ─────────────────────────────────────────────────────────────

@app.get("/api/memory/{student_id}")
async def get_memory(student_id: str):
    try:
        memory = await db.router_memory.get(student_id)
    except Exception as e:
        raise HTTPException(500, f"MongoDB error: {e}")
    if not memory:
        raise HTTPException(404, f"No memory found for student '{student_id}'")
    return memory


# ═════════════════════════════════════════════════════════════════
# STRATEGY 15: PRODUCTION LAUNCH CONFIGURATION
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn

    log.info("Starting Dr. Mind v6.0 Production Server")
    log.info(f"  Host: {Config.HOST}:{Config.PORT}")
    log.info(f"  Workers: {Config.WORKERS}")

    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        workers=Config.WORKERS,
        log_level="info",
        access_log=True,
        timeout_keep_alive=30,
        limit_concurrency=Config.MAX_CONCURRENT + 10,
        limit_max_requests=10000,
    )