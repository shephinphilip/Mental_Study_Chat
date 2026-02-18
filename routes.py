#!/usr/bin/env python3
"""
Dr. Mind v5.1 — FastAPI Backend
Converts the CLI application into a REST API.

Endpoints:
  POST /api/session/start          — initialise agent + session for a student
  POST /api/chat                   — send a message, receive clinical response
  GET  /api/history/{student_id}   — last N encounters from DB
  POST /api/session/clear          — wipe conversation history (keep session)
  GET  /api/profiles               — list available patient profiles
  GET  /api/health                 — liveness check

Run:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import sqlite3
import hashlib
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from models import ClinicalEssentials, RiskLevel, MOCK_STUDENTS
from crisis import build_agent

load_dotenv()

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dr_mind")

# ─── GPT-4o pricing (USD per token) ──────────────────────────────────────────
# https://openai.com/pricing  (gpt-4o as of 2024-11)
GPT4O_INPUT_COST_PER_TOKEN  = 5.00  / 1_000_000   # $5.00  / 1M input tokens
GPT4O_OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000   # $15.00 / 1M output tokens


# ─── Token tracker callback ───────────────────────────────────────────────────

class TokenTracker(BaseCallbackHandler):
    """
    Accumulates token usage across ALL LangChain LLM calls made during a
    single agent.invoke().  Works with LangGraph because LangGraph honours
    the callbacks passed in config.
    """

    def __init__(self):
        self.llm_calls:          int   = 0
        self.prompt_tokens:      int   = 0
        self.completion_tokens:  int   = 0
        self._call_log:          list  = []   # per-call breakdown

    # LangChain fires this after every LLM response
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.llm_calls += 1
        usage = (response.llm_output or {}).get("token_usage", {})
        pt = usage.get("prompt_tokens",     0)
        ct = usage.get("completion_tokens", 0)
        self.prompt_tokens     += pt
        self.completion_tokens += ct
        self._call_log.append({"call": self.llm_calls, "prompt": pt, "completion": ct})

    # ── Derived metrics ──────────────────────────────────────────────────────

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return (self.prompt_tokens      * GPT4O_INPUT_COST_PER_TOKEN +
                self.completion_tokens  * GPT4O_OUTPUT_COST_PER_TOKEN)

    def summary(self) -> dict:
        return {
            "llm_calls":         self.llm_calls,
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.total_tokens,
            "cost_usd":          round(self.cost_usd, 6),
            "per_call_detail":   self._call_log,
        }

# ─── App setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Dr. Mind Clinical API",
    description="Psychiatric interview engine for Indian teenagers",
    version="5.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_PATH = "dr_mind_clinical.db"

# ─── In-memory session store ──────────────────────────────────────────────────
# Key: session_id (str)
# Value: {
#   "agent":    compiled LangGraph agent,
#   "history":  List[HumanMessage | AIMessage],
#   "profile":  ClinicalEssentials,
#   "storage":  ClinicalStorage,
#   "created":  datetime,
# }
SESSIONS: Dict[str, dict] = {}


# ─── Database ─────────────────────────────────────────────────────────────────

@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clinical_encounters (
                encounter_id          INTEGER PRIMARY KEY,
                session_id            TEXT NOT NULL,
                student_id            TEXT NOT NULL,
                timestamp             TEXT NOT NULL,
                classification        TEXT,
                clinical_reasoning    TEXT,
                risk_level_at_time    TEXT,
                user_message          TEXT,
                ai_response           TEXT,
                crisis_flags_triggered TEXT,
                immediate_action_taken TEXT
            )
        """)
        # safe migration for existing DBs
        for col in ["clinical_reasoning", "ai_response"]:
            try:
                conn.execute(f"ALTER TABLE clinical_encounters ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass
        conn.commit()


class ClinicalStorage:
    def __init__(self, student_id: str, session_id: str):
        self.student_id = student_id
        self.session_id = session_id

    def record(
        self,
        classification: str,
        reasoning: str,
        risk_level: str,
        user_message: str,
        ai_response: str,
        crisis_flags: List[str],
        action_taken: str,
    ):
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO clinical_encounters
                  (session_id, student_id, timestamp, classification, clinical_reasoning,
                   risk_level_at_time, user_message, ai_response,
                   crisis_flags_triggered, immediate_action_taken)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    self.session_id,
                    self.student_id,
                    datetime.now().isoformat(),
                    classification,
                    reasoning[:500] if reasoning else "",
                    risk_level,
                    user_message[:500],
                    ai_response[:1000],
                    json.dumps(crisis_flags),
                    action_taken,
                ),
            )
            conn.commit()

    def get_recent(self, limit: int = 10) -> List[dict]:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, classification, risk_level_at_time,
                       user_message, ai_response, immediate_action_taken
                FROM clinical_encounters
                WHERE student_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (self.student_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or "your-key" in api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    init_database()
    log.info("✅ Dr. Mind API ready")


# ─── Pydantic request/response schemas ───────────────────────────────────────

class StartSessionRequest(BaseModel):
    student_id: str


class StartSessionResponse(BaseModel):
    session_id:   str
    student_id:   str
    name:         str
    risk_level:   str
    age:          int
    chief_concern: str


class ChatRequest(BaseModel):
    session_id: str
    message:    str


class UsageInfo(BaseModel):
    llm_calls:         int
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int
    cost_usd:          float
    per_call_detail:   List[dict]


class ChatResponse(BaseModel):
    response:           str
    classification:     str
    inquiry_stage:      str
    session_id:         str
    detected_language:  str
    usage:              UsageInfo


class ClearRequest(BaseModel):
    session_id: str


class EncounterRecord(BaseModel):
    timestamp:             str
    classification:        str
    risk_level_at_time:    str
    user_message:          str
    ai_response:           Optional[str]
    immediate_action_taken: Optional[str]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "sessions_active": len(SESSIONS)}


@app.get("/api/profiles")
def list_profiles():
    """Return all available patient profiles (for the UI selector)."""
    result = []
    for sid, profile in MOCK_STUDENTS.items():
        result.append({
            "student_id":   sid,
            "name":         profile.name,
            "age":          profile.age,
            "current_class": profile.current_class,
            "city":         profile.city,
            "risk_level":   profile.current_risk_level.value,
            "chief_concern": profile.chief_concern,
        })
    return result


@app.post("/api/session/start", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest):
    """
    Initialise a new session for the given student_id.
    Builds a fresh LangGraph agent with the student's profile.
    Returns a session_id that must be passed to /api/chat.
    """
    profile = MOCK_STUDENTS.get(req.student_id)
    if not profile:
        raise HTTPException(404, f"Student '{req.student_id}' not found")

    # Build deterministic-but-unique session id
    session_id = hashlib.md5(
        f"{req.student_id}_{datetime.now().isoformat()}".encode()
    ).hexdigest()[:16]

    agent   = build_agent(profile)
    storage = ClinicalStorage(profile.student_id, session_id)

    SESSIONS[session_id] = {
        "agent":   agent,
        "history": [],
        "profile": profile,
        "storage": storage,
        "created": datetime.now(),
    }

    return StartSessionResponse(
        session_id=session_id,
        student_id=profile.student_id,
        name=profile.name,
        risk_level=profile.current_risk_level.value,
        age=profile.age,
        chief_concern=profile.chief_concern,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Send a student message and receive Dr. Mind's response.
    Conversation history is maintained server-side per session_id.
    """
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found. Call /api/session/start first.")

    message = req.message.strip()
    if not message:
        raise HTTPException(400, "Message cannot be empty.")

    agent   = session["agent"]
    history = session["history"]
    profile = session["profile"]
    storage = session["storage"]

    # Append user message to history
    history.append(HumanMessage(content=message))

    # Invoke the LangGraph agent with TokenTracker attached
    prev_language = session.get("detected_language", "ENGLISH")
    tracker = TokenTracker()
    result = agent.invoke(
        {
            "messages":           history,
            "classification":     "SAFE",
            "clinical_reasoning": "",
            "age_estimate":       profile.get_age_bracket(),
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
    inquiry_stage     = result.get("inquiry_stage", "initial")
    detected_language = result.get("detected_language", prev_language)

    # ── Token / cost summary ─────────────────────────────────────────────────
    usage = tracker.summary()
    log.info(
        "💬 QUERY  student=%-10s  cls=%-20s  lang=%s",
        profile.student_id, classification, detected_language,
    )
    log.info(
        "📊 USAGE  llm_calls=%d  prompt=%d  completion=%d  total=%d  cost=$%.6f",
        usage["llm_calls"],
        usage["prompt_tokens"],
        usage["completion_tokens"],
        usage["total_tokens"],
        usage["cost_usd"],
    )
    for c in usage["per_call_detail"]:
        log.info(
            "   ↳ call #%d  prompt=%d  completion=%d",
            c["call"], c["prompt"], c["completion"],
        )

    # Persist detected language in session for next turn
    session["detected_language"] = detected_language

    # Append AI response to history
    history.append(AIMessage(content=response_text))

    # Trim history to last 20 messages to control token growth
    if len(history) > 20:
        session["history"] = history[-20:]
    else:
        session["history"] = history

    # Persist to DB
    storage.record(
        classification=classification,
        reasoning=result.get("clinical_reasoning", ""),
        risk_level=profile.current_risk_level.value,
        user_message=message,
        ai_response=response_text,
        crisis_flags=[],
        action_taken=f"stage:{inquiry_stage}",
    )

    return ChatResponse(
        response=response_text,
        classification=classification,
        inquiry_stage=inquiry_stage,
        session_id=req.session_id,
        detected_language=detected_language,
        usage=UsageInfo(**usage),
    )


@app.post("/api/session/clear")
def clear_session(req: ClearRequest):
    """Wipe conversation history for a session (keeps the session alive)."""
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found.")
    session["history"] = []
    return {"status": "cleared", "session_id": req.session_id}


@app.get("/api/history/{student_id}", response_model=List[EncounterRecord])
def get_history(student_id: str, limit: int = 10):
    """Return the last N clinical encounters for a student from the DB."""
    if student_id not in MOCK_STUDENTS:
        raise HTTPException(404, f"Student '{student_id}' not found")
    storage = ClinicalStorage(student_id, "")
    records = storage.get_recent(limit=limit)
    return records


# ─── Dev runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)