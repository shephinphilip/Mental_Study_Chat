#!/usr/bin/env python3
"""
Dr. Mind v5.3 — main.py
Full FastAPI application: MongoDB (zenark) + Meditation Engine + Conversation History

Endpoints:
  ── Session ──────────────────────────────────────────────────────
  POST /api/session/start            Start a new clinical session
  POST /api/session/clear            Clear in-memory history (keep session alive)
  GET  /api/profiles                 List available student profiles

  ── Chat ─────────────────────────────────────────────────────────
  POST /api/chat                     Send a message, receive clinical response
                                     → includes meditation suggestion if appropriate

  ── Conversation History ──────────────────────────────────────────
  POST /api/conversations/save       Explicitly persist a full session to MongoDB
  GET  /api/conversations/{user_id}  All saved sessions for a user
  GET  /api/conversations/{user_id}/messages  Flat per-message history (analytics)
  DELETE /api/conversations/{session_id}      Delete a session

  ── Meditations ──────────────────────────────────────────────────
  GET  /api/meditations              List all meditations
  GET  /api/meditations/suggest/{classification}  Suggest for a clinical category
  POST /api/meditations/progress/start  Log session start
  POST /api/meditations/progress/complete  Mark session complete
  GET  /api/meditations/streaks/{user_id}  Get meditation streak data
  GET  /api/meditations/audios       Fetch active audio tracks

  ── Router Memory ────────────────────────────────────────────────
  GET  /api/memory/{student_id}      Get student's router memory (persistent state)

  ── Health ───────────────────────────────────────────────────────
  GET  /api/health                   Liveness + DB connectivity check

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import LLMResult
from pydantic import BaseModel

from crisis import build_agent, MEDITATION_MAP, MEDITATIONS
from database import db

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dr_mind")

# ── Pricing ───────────────────────────────────────────────────────────────────
# gpt-4o-mini (classifier):  $0.15 / 1M input,  $0.60 / 1M output
# gpt-4o      (responder):   $5.00 / 1M input,  $15.00 / 1M output
MINI_IN  = 0.15  / 1_000_000
MINI_OUT = 0.60  / 1_000_000
GPT4O_IN = 5.00  / 1_000_000
GPT4O_OUT= 15.00 / 1_000_000


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
        pt = usage.get("prompt_tokens",    0)
        ct = usage.get("completion_tokens", 0)
        self.prompt_tokens     += pt
        self.completion_tokens += ct
        self._call_log.append({"call": self.llm_calls, "prompt": pt, "completion": ct})

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        # Approximate blended: first call is mini, subsequent is gpt-4o
        if self.llm_calls == 0:
            return 0.0
        # Classifier call (mini) — assume 20 completion tokens
        mini_est_pt = min(self.prompt_tokens, 900)
        mini_est_ct = 20
        remaining_pt = max(0, self.prompt_tokens - mini_est_pt)
        remaining_ct = max(0, self.completion_tokens - mini_est_ct)
        return (
            mini_est_pt  * MINI_IN  + mini_est_ct  * MINI_OUT +
            remaining_pt * GPT4O_IN + remaining_ct * GPT4O_OUT
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
# {session_id: {agent, history, profile, detected_language, created, mongo_saved}}
SESSIONS: Dict[str, dict] = {}


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Dr. Mind Clinical API",
    description="Psychiatric interview engine for Indian teenagers — zenark backend",
    version="5.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or "your-key" in api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")

    mongo_ok = await db.ping()
    if not mongo_ok:
        log.warning("⚠️  MongoDB not reachable — check MONGODB_URI in .env")
    else:
        log.info("✅ MongoDB (zenark) connected")

    log.info("✅ Dr. Mind v5.3 API ready")


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    student_id: str
    user_id:    Optional[str] = None   # MongoDB ObjectId of logged-in user


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
    user_id:    Optional[str] = None   # required for MongoDB persistence


class MeditationSuggestion(BaseModel):
    id:          int
    title:       str
    description: str
    relevance:   str    # why it's suggested for this classification


class UsageInfo(BaseModel):
    llm_calls:         int
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int
    cost_usd:          float
    per_call_detail:   List[dict]


class ChatResponse(BaseModel):
    response:             str
    classification:       str
    inquiry_stage:        str
    session_id:           str
    detected_language:    str
    usage:                UsageInfo
    meditation_suggestion: Optional[MeditationSuggestion] = None


class ClearRequest(BaseModel):
    session_id: str


class SaveConversationRequest(BaseModel):
    session_id: str
    user_id:    str


class MeditationProgressStartRequest(BaseModel):
    user_id:    str
    meditation_id: int   # corresponds to MEDITATIONS dict id


class MeditationProgressCompleteRequest(BaseModel):
    user_id:       str
    meditation_id: int
    time_spent:    int   # seconds


# ── Helper: convert LangChain messages to serialisable dicts ──────────────────

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
    """Save / update the session in MongoDB chats + normalized collections."""
    msgs_raw  = _msgs_to_dicts(session["history"])
    # 1. Upsert full chat document
    await db.chats.upsert_session(
        session_id=session_id,
        user_id=user_id,
        messages=msgs_raw,
    )
    log.info("💾 session %s persisted to MongoDB chats (%d msgs)", session_id, len(msgs_raw))


async def _append_normalized(
    student_id:    str,
    speaker:       str,
    text:          str,
    classification: str,
) -> None:
    """Append one message to the flat analytics collection."""
    await db.normalized.append_message(
        student_id=student_id,
        speaker=speaker,
        message_text=text,
        labels=[classification],
    )


# ── MOCK_STUDENTS import (from models) ────────────────────────────────────────
# Keep the existing profile system working alongside MongoDB users.
try:
    from models import MOCK_STUDENTS
except ImportError:
    MOCK_STUDENTS = {}
    log.warning("⚠️  models.py not found — /api/profiles will return empty list")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    mongo_ok = await db.ping()
    return {
        "status":          "ok",
        "sessions_active": len(SESSIONS),
        "mongodb":         "connected" if mongo_ok else "unreachable",
        "version":         "5.3.0",
    }


# ── Profiles ─────────────────────────────────────────────────────────────────

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
        "classifications":   [],    # track per-turn categories for router memory
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


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found. Call /api/session/start first.")

    message = req.message.strip()
    if not message:
        raise HTTPException(400, "Message cannot be empty.")

    agent   = session["agent"]
    history = session["history"]
    profile = session["profile"]
    user_id = req.user_id or session.get("user_id", profile.student_id)

    history.append(HumanMessage(content=message))

    prev_language = session.get("detected_language", "ENGLISH")
    tracker       = TokenTracker()

    result = agent.invoke(
        {
            "messages":           history,
            "classification":     "SAFE",
            "clinical_reasoning": "",
            "age_estimate":       profile.get_age_bracket() if hasattr(profile, "get_age_bracket") else "teen",
            "inquiry_stage":      "initial",
            "identified_drivers": [],
            "current_input":      message,
            "crisis_verified":    False,
            "detected_language":  prev_language,
        },
        config={"callbacks": [tracker]},
    )

    response_text     = result["messages"][-1].content
    classification    = result.get("classification", "SAFE")
    inquiry_stage     = result.get("inquiry_stage",  "initial")
    detected_language = result.get("detected_language", prev_language)
    usage             = tracker.summary()

    # ── Log usage ────────────────────────────────────────────────────────────
    log.info(
        "💬 QUERY  student=%-10s  cls=%-20s  lang=%s",
        profile.student_id, classification, detected_language,
    )
    log.info(
        "📊 USAGE  llm_calls=%d  prompt=%d  completion=%d  total=%d  cost=$%.6f",
        usage["llm_calls"], usage["prompt_tokens"],
        usage["completion_tokens"], usage["total_tokens"], usage["cost_usd"],
    )
    for c in usage["per_call_detail"]:
        log.info("   ↳ call #%d  prompt=%d  completion=%d", c["call"], c["prompt"], c["completion"])

    # ── Meditation suggestion ────────────────────────────────────────────────
    suggestion = _get_meditation_suggestion(classification)

    # ── Update session ────────────────────────────────────────────────────────
    session["detected_language"] = detected_language
    history.append(AIMessage(content=response_text))
    session["history"]           = history[-20:]   # rolling window
    session["classifications"].append(classification)

    # ── MongoDB — append normalized messages ──────────────────────────────────
    try:
        await _append_normalized(profile.student_id, "student",  message,       classification)
        await _append_normalized(profile.student_id, "dr_mind",  response_text, classification)
    except Exception as e:
        log.warning("⚠️  normalized append failed: %s", e)

    # ── MongoDB — upsert router memory ────────────────────────────────────────
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

    # ── MongoDB — auto-persist full chat every 5 turns ────────────────────────
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
    )


# ── Conversation History ──────────────────────────────────────────────────────

@app.post("/api/conversations/save")
async def save_conversation(req: SaveConversationRequest):
    """Explicitly persist the current session to MongoDB."""
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
    """Return all saved chat sessions for a user (summaries, newest first)."""
    try:
        sessions = await db.chats.get_user_sessions(user_id, limit=limit)
    except Exception as e:
        raise HTTPException(500, f"MongoDB error: {e}")

    # Slim down: don't return full message bodies in the list view
    result = []
    for s in sessions:
        msgs = s.get("messages", [])
        result.append({
            "session_id":   s.get("session_id"),
            "timestamp":    str(s.get("timestamp", "")),
            "message_count": len(msgs),
            "preview":      msgs[0]["content"][:80] if msgs else "",
        })
    return result


@app.get("/api/conversations/{user_id}/messages")
async def get_conversation_messages(
    user_id: str,
    session_id: Optional[str] = Query(default=None),
    limit:   int = Query(default=50, ge=1, le=200),
):
    """
    Flat per-message history from chat_sessions_normalized.
    Optionally filter to a specific session_id.
    """
    try:
        messages = await db.normalized.get_student_history(user_id, limit=limit)
    except Exception as e:
        raise HTTPException(500, f"MongoDB error: {e}")

    return {"student_id": user_id, "messages": messages, "count": len(messages)}


@app.get("/api/conversations/session/{session_id}")
async def get_session_detail(session_id: str):
    """Return full message list for a single saved session."""
    try:
        session = await db.chats.get_session(session_id)
    except Exception as e:
        raise HTTPException(500, f"MongoDB error: {e}")

    if not session:
        raise HTTPException(404, "Session not found in MongoDB.")
    return session


@app.delete("/api/conversations/{session_id}")
async def delete_conversation(session_id: str):
    """Delete a saved session from MongoDB."""
    try:
        deleted = await db.chats.delete_session(session_id)
    except Exception as e:
        raise HTTPException(500, f"MongoDB error: {e}")

    if not deleted:
        raise HTTPException(404, "Session not found.")

    # Also remove from in-memory store if present
    SESSIONS.pop(session_id, None)
    return {"status": "deleted", "session_id": session_id}


# ── Meditations ───────────────────────────────────────────────────────────────

@app.get("/api/meditations")
async def list_meditations():
    """Return all available meditation sessions."""
    return {"meditations": list(MEDITATIONS.values()), "count": len(MEDITATIONS)}


@app.get("/api/meditations/suggest/{classification}")
async def suggest_meditation(classification: str):
    """Return the best-fit meditation(s) for a clinical classification."""
    suggestion = _get_meditation_suggestion(classification.upper())
    if not suggestion:
        raise HTTPException(404, f"No meditation mapping for '{classification}'")
    return suggestion


@app.post("/api/meditations/progress/start")
async def meditation_start(req: MeditationProgressStartRequest):
    """Log the start of a meditation session."""
    med = MEDITATIONS.get(req.meditation_id)
    if not med:
        raise HTTPException(404, f"Meditation id {req.meditation_id} not found")
    try:
        session_id = f"med_{req.user_id}_{req.meditation_id}_{int(datetime.now().timestamp())}"
        doc_id     = await db.meditation_progress.start(req.user_id, session_id)
    except Exception as e:
        raise HTTPException(500, f"MongoDB error: {e}")

    return {"status": "started", "progress_id": doc_id, "meditation": med["title"]}


@app.post("/api/meditations/progress/complete")
async def meditation_complete(req: MeditationProgressCompleteRequest):
    """Mark a meditation session complete and update streak."""
    med = MEDITATIONS.get(req.meditation_id)
    if not med:
        raise HTTPException(404, f"Meditation id {req.meditation_id} not found")

    session_id = f"med_{req.user_id}_{req.meditation_id}"   # approximate match
    try:
        await db.meditation_progress.complete(req.user_id, session_id, req.time_spent)
        streak = await db.meditation_streaks.update_after_session(req.user_id)
    except Exception as e:
        raise HTTPException(500, f"MongoDB error: {e}")

    return {
        "status":        "completed",
        "meditation":    med["title"],
        "time_spent_s":  req.time_spent,
        "streak":        {
            "current":  streak.get("current_streak"),
            "longest":  streak.get("longest_streak"),
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
    """Return active audio tracks, optionally filtered by genre."""
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
    """Return the persistent router memory for a student."""
    try:
        memory = await db.router_memory.get(student_id)
    except Exception as e:
        raise HTTPException(500, f"MongoDB error: {e}")
    if not memory:
        raise HTTPException(404, f"No memory found for student '{student_id}'")
    return memory


# ── Meditation suggestion helper ──────────────────────────────────────────────

def _get_meditation_suggestion(classification: str) -> Optional[MeditationSuggestion]:
    """Return the primary meditation suggestion for a clinical classification."""
    entry = MEDITATION_MAP.get(classification)
    if not entry:
        return None
    primary_id = entry["ids"][0]
    med        = MEDITATIONS.get(primary_id)
    if not med:
        return None
    return MeditationSuggestion(
        id=primary_id,
        title=med["title"],
        description=med["description"],
        relevance=entry["relevance"],
    )


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)