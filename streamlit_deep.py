"""
Zenarck Deep Agent — Production-grade Streamlit frontend
Mental Health Support AI | Behavioral Intelligence Platform
Version: 2.0.1
Backend: https://mental-study-chat.onrender.com
"""

import streamlit as st
import requests
import uuid
import json
import time
import os
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# ─────────────────────────────────────────────
# PAGE CONFIG — must be FIRST Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Zenarck Deep Agent",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Zenarck Deep Agent — Mental Health Support AI v2.0",
    },
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
# Priority order for backend URL resolution:
#   1. Streamlit Secrets  → [backend] url
#   2. Environment var    → BACKEND_URL
#   3. Hardcoded Render   → mental-study-chat.onrender.com  (production default)
RENDER_BACKEND = "https://mental-study-chat.onrender.com"

def _get_backend_url() -> str:
    try:
        url = st.secrets["backend"]["url"].strip().rstrip("/")
        if url:
            return url
    except Exception:
        pass
    env = os.environ.get("BACKEND_URL", "").strip().rstrip("/")
    if env:
        return env
    return RENDER_BACKEND

BACKEND_URL: str = _get_backend_url()
API_BASE: str = f"{BACKEND_URL}/api/v1"

# Render free tier spins down after 15 min of inactivity — first request
# can take 30-50 s to cold-start. Timeouts are raised to cover this.
REQUEST_TIMEOUT  = 90   # seconds — covers Render cold-start + LLM call
SESSION_TIMEOUT  = 60   # seconds — session/start probe also hits cold instance
HEALTH_TIMEOUT   = 12   # seconds — health check is lighter
MAX_RETRIES      = 2

CRISIS_SCENARIOS = {"crisis"}
URGENT_SCENARIOS = {"negative", "exam_stress"}

SCENARIO_LABELS: Dict[str, str] = {
    "crisis":      "🚨 Crisis",
    "negative":    "💙 Distress",
    "exam_stress": "📚 Exam Stress",
    "positive":    "✨ Positive",
    "marks":       "📊 Marks",
    "generic":     "💬 General",
    "mischievous": "⚠️ Off-topic",
    "unknown":     "❓ Unknown",
}

HELPLINES = """**Emergency Helplines (India)**
- AASRA: 91-22-27546669
- Vandrevala Foundation: 1860-2662-345
- iCall: 9152987821
- Emergency: **112**"""

# ─────────────────────────────────────────────
# THEME & CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #0d0f14;
    --surface:   #141720;
    --surface2:  #1c2030;
    --border:    rgba(255,255,255,0.07);
    --accent:    #4f9cf9;
    --accent2:   #8b5cf6;
    --crisis:    #ef4444;
    --warn:      #f59e0b;
    --ok:        #10b981;
    --text:      #e8ecf4;
    --muted:     #7e8a9e;
    --radius:    12px;
    --font:      'DM Sans', sans-serif;
    --font-head: 'DM Serif Display', serif;
}

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Main container ── */
.block-container {
    max-width: 780px !important;
    padding: 2rem 1.5rem 6rem !important;
}

/* ── Page header ── */
.zn-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.zn-logo {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
    box-shadow: 0 0 20px rgba(79,156,249,0.3);
}
.zn-title { font-family: var(--font-head); font-size: 1.5rem; color: var(--text); line-height: 1.2; }
.zn-subtitle { font-size: 0.78rem; color: var(--muted); margin-top: 2px; letter-spacing: 0.04em; text-transform: uppercase; }

/* ── Status pill ── */
.zn-status {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.76rem; font-weight: 500; letter-spacing: 0.02em;
}
.zn-status-ok   { background: rgba(16,185,129,0.12); color: var(--ok);   border: 1px solid rgba(16,185,129,0.25); }
.zn-status-warn { background: rgba(245,158,11,0.12); color: var(--warn); border: 1px solid rgba(245,158,11,0.25); }
.zn-status-err  { background: rgba(239,68,68,0.12);  color: var(--crisis); border: 1px solid rgba(239,68,68,0.25); }

/* ── Chat bubbles ── */
.zn-msg { margin-bottom: 1.25rem; display: flex; flex-direction: column; }
.zn-msg-user   { align-items: flex-end; }
.zn-msg-agent  { align-items: flex-start; }

.zn-bubble {
    max-width: 88%;
    padding: 14px 18px;
    border-radius: var(--radius);
    font-size: 0.95rem;
    line-height: 1.65;
    word-break: break-word;
    position: relative;
}
.zn-bubble-user {
    background: linear-gradient(135deg, rgba(79,156,249,0.2), rgba(139,92,246,0.15));
    border: 1px solid rgba(79,156,249,0.25);
    color: var(--text);
    border-bottom-right-radius: 4px;
}
.zn-bubble-agent {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    border-bottom-left-radius: 4px;
}
.zn-bubble-crisis {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.35) !important;
}
.zn-bubble-meta {
    font-size: 0.72rem; color: var(--muted);
    margin-top: 5px; padding: 0 4px;
    display: flex; align-items: center; gap: 8px;
}
.zn-scenario-tag {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 0.70rem; font-weight: 500;
    padding: 2px 8px; border-radius: 20px;
    background: rgba(255,255,255,0.05); border: 1px solid var(--border);
}

/* ── Typing indicator ── */
.zn-typing { display: flex; align-items: center; gap: 6px; padding: 12px 0; }
.zn-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.2s ease-in-out infinite;
}
.zn-dot:nth-child(2) { animation-delay: 0.2s; background: var(--accent2); }
.zn-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse {
    0%,100% { opacity: 0.3; transform: scale(0.8); }
    50%      { opacity: 1;   transform: scale(1.2); }
}

/* ── Crisis banner ── */
.zn-crisis-banner {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: var(--radius);
    padding: 14px 18px;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    line-height: 1.6;
}

/* ── Info cards ── */
.zn-info-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin-bottom: 1rem;
}
.zn-info-card h4 {
    font-size: 0.82rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--muted); margin: 0 0 10px;
}

/* ── Buttons ── */
[data-testid="baseButton-primary"] button,
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important;
    font-family: var(--font) !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Inputs ── */
[data-testid="stChatInput"] textarea,
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: var(--font) !important;
}
[data-testid="stChatInput"] { background: var(--surface) !important; border-top: 1px solid var(--border) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stExpander"] summary { color: var(--muted) !important; font-size: 0.82rem !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }

/* ── Checkbox & toggles ── */
[data-testid="stCheckbox"] label { color: var(--text) !important; font-size: 0.88rem !important; }

/* ── Latency badge ── */
.zn-latency {
    font-size: 0.70rem; color: var(--muted);
    font-variant-numeric: tabular-nums;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages":        [],       # list of {role, content, metadata, ts}
        "session_key":     None,     # active session key from backend
        "user_id":         None,     # stable anonymous user ID
        "backend_status":  "checking",  # "ok" | "degraded" | "down" | "checking"
        "turn_count":      0,
        "show_debug":      False,
        "crisis_active":   False,
        "session_started_at": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.user_id is None:
        # Stable ID seeded from browser session — random fallback
        raw = str(uuid.uuid4())
        st.session_state.user_id = hashlib.sha256(raw.encode()).hexdigest()[:12]

_init_state()

# ─────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────
def _headers() -> Dict[str, str]:
    """Build request headers. Inject API key if configured in secrets."""
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        key = st.secrets["backend"]["api_key"]
        if key:
            h["X-API-Key"] = key
    except Exception:
        pass
    return h


def _check_backend_health() -> str:
    """Quick health probe. Returns 'ok' | 'degraded' | 'down'."""
    if not API_BASE:
        return "down"
    try:
        r = requests.get(f"{BACKEND_URL}/health", headers=_headers(), timeout=HEALTH_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            return "ok" if data.get("ready", False) else "degraded"
        return "degraded"
    except Exception:
        return "down"


def _start_session(user_id: str) -> Optional[Dict[str, Any]]:
    """Create a new session on the backend. Returns parsed JSON or None."""
    if not API_BASE:
        return None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                f"{API_BASE}/session/start",
                json={"user_id": user_id},
                headers=_headers(),
                timeout=SESSION_TIMEOUT,
            )
            if r.status_code == 200:
                return r.json()
            st.warning(f"Session start failed ({r.status_code}): {r.text[:200]}")
            return None
        except requests.exceptions.ConnectionError:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(0.5)
        except Exception as e:
            st.warning(f"Session error: {e}")
            return None
    return None


def _send_message(session_key: str, message: str) -> Optional[Dict[str, Any]]:
    """Send a chat message. Returns parsed response dict or None."""
    if not API_BASE:
        return None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                f"{API_BASE}/chat/deep",
                json={"session_key": session_key, "message": message},
                headers=_headers(),
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                return {"_error": "rate_limit", "detail": "Server busy — please wait a moment and try again."}
            return {"_error": "api_error", "detail": f"[{r.status_code}] {r.text[:300]}"}
        except requests.exceptions.Timeout:
            if attempt == MAX_RETRIES - 1:
                return {"_error": "timeout", "detail": "The server is taking too long. Please try again."}
            time.sleep(1)
        except requests.exceptions.ConnectionError:
            return {"_error": "connection", "detail": "Cannot reach the backend. Check your connection."}
        except Exception as e:
            return {"_error": "unknown", "detail": str(e)}
    return None

# ─────────────────────────────────────────────
# RENDERING HELPERS
# ─────────────────────────────────────────────
def _scenario_tag(scenario: Optional[str]) -> str:
    label = SCENARIO_LABELS.get(scenario or "", f"💬 {scenario or 'unknown'}")
    return f'<span class="zn-scenario-tag">{label}</span>'


def _render_message(msg: Dict[str, Any]):
    role = msg["role"]
    content = msg["content"]
    meta = msg.get("metadata", {})
    ts = msg.get("ts", "")
    scenario = meta.get("detected_scenario", "")
    is_crisis = scenario in CRISIS_SCENARIOS
    latency = meta.get("latency_seconds")

    if role == "user":
        bubble_class = "zn-bubble-user"
        row_class = "zn-msg-user"
    else:
        bubble_class = "zn-bubble-agent" + (" zn-bubble-crisis" if is_crisis else "")
        row_class = "zn-msg-agent"

    # Build content lines preserving newlines
    safe_content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")

    tag_html = _scenario_tag(scenario) if (role == "assistant" and scenario) else ""
    latency_html = f'<span class="zn-latency">⏱ {latency:.2f}s</span>' if latency and role == "assistant" else ""
    ts_html = f'<span>{ts}</span>' if ts else ""

    meta_row = ""
    if role == "assistant":
        meta_row = f'<div class="zn-bubble-meta">{tag_html}{latency_html}{ts_html}</div>'

    st.markdown(
        f"""
        <div class="zn-msg {row_class}">
            <div class="zn-bubble {bubble_class}">{safe_content}</div>
            {meta_row}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Debug expander
    if st.session_state.show_debug and role == "assistant" and meta:
        with st.expander("🔍 Internal Thought Process", expanded=False):
            st.json(meta)


def _render_typing():
    st.markdown(
        '<div class="zn-typing"><div class="zn-dot"></div><div class="zn-dot"></div><div class="zn-dot"></div></div>',
        unsafe_allow_html=True,
    )


def _status_pill(status: str) -> str:
    if status == "ok":
        return '<span class="zn-status zn-status-ok">● Backend online</span>'
    if status == "degraded":
        return '<span class="zn-status zn-status-warn">⚠ Backend degraded</span>'
    if status == "checking":
        return '<span class="zn-status zn-status-warn">○ Checking backend…</span>'
    return '<span class="zn-status zn-status-err">✕ Backend offline (may be waking up)</span>'


def _render_coldstart_notice():
    """Show a subtle notice that Render may be waking up."""
    st.markdown(
        """
        <div class="zn-info-card" style="border-color:rgba(245,158,11,0.25);">
        <h4>⏳ Backend Waking Up</h4>
        <p style="font-size:0.88rem;color:#e8ecf4;margin:0;line-height:1.6;">
        The backend is hosted on Render's free tier and may take
        <strong>30–50 seconds</strong> to respond after a period of inactivity.
        Your first message will arrive — just give it a moment.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Session")
    st.caption(f"Backend: `mental-study-chat.onrender.com`")

    # Backend health check (cached for 30s via session state)
    health_last_check = st.session_state.get("health_last_check", 0)
    if time.time() - health_last_check > 30:
        st.session_state.backend_status = _check_backend_health()
        st.session_state.health_last_check = time.time()

    status = st.session_state.backend_status or "down"
    st.markdown(_status_pill(status), unsafe_allow_html=True)

    st.divider()

    # Anonymous user ID display
    uid = st.session_state.user_id or "—"
    st.caption(f"User ID: `{uid[:12]}`")

    if st.session_state.session_key:
        sk = st.session_state.session_key
        st.markdown(
            f'<div class="zn-info-card"><h4>Active Session</h4>'
            f'<code style="font-size:0.78rem;word-break:break-all;">{sk}</code>'
            f'<br><span style="font-size:0.76rem;color:#7e8a9e;">Turn {st.session_state.turn_count}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:0.84rem;color:#f59e0b;margin:8px 0;">No active session.<br>Click below to start.</div>',
            unsafe_allow_html=True,
        )

    btn_label = "🔄 New Session" if st.session_state.session_key else "▶ Start Session"
    if st.button(btn_label, type="primary", use_container_width=True):
        with st.spinner("Connecting… (may take up to 30s on cold start)"):
            data = _start_session(st.session_state.user_id)
        if data and data.get("session_key"):
            st.session_state.session_key = data["session_key"]
            st.session_state.messages = []
            st.session_state.turn_count = 0
            st.session_state.crisis_active = False
            st.session_state.session_started_at = datetime.now(timezone.utc).strftime("%H:%M UTC")
            st.success("Session started!")
            st.rerun()
        else:
            st.error("Could not start session. Backend may still be waking up — try again in 30s.")

    st.divider()

    st.markdown("### 🛠 Options")
    st.session_state.show_debug = st.checkbox(
        "Show internal thought process",
        value=st.session_state.show_debug,
        help="Reveals detected scenario, language, and latency for each response."
    )

    if st.session_state.messages:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.crisis_active = False
            st.rerun()

    st.divider()
    st.markdown(
        '<div style="font-size:0.72rem;color:#7e8a9e;line-height:1.6;">'
        'Zenarck Deep Agent v2.0<br>'
        'Mental Health Support AI<br>'
        '© 2025 Zenarck</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────

# ── Header ──
st.markdown(
    """
    <div class="zn-header">
        <div class="zn-logo">🧠</div>
        <div>
            <div class="zn-title">Zenarck Deep Agent</div>
            <div class="zn-subtitle">Mental Health Support AI · Behavioral Intelligence</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Render cold-start notice (shown when backend is down / first load) ──
if st.session_state.backend_status == "down":
    _render_coldstart_notice()

# ── Crisis banner (persists while crisis active) ──
if st.session_state.crisis_active:
    st.markdown(
        f'<div class="zn-crisis-banner">🚨 <strong>Crisis mode active.</strong> '
        f'If you are in immediate danger, please reach out right now.<br><br>'
        f'{HELPLINES.replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )

# ── Welcome card (no session) ──
if not st.session_state.messages and not st.session_state.session_key:
    st.markdown(
        """
        <div class="zn-info-card" style="margin-top:1rem;">
            <h4>👋 Welcome</h4>
            <p style="font-size:0.9rem;color:#e8ecf4;line-height:1.65;margin:0;">
            This is a safe, confidential space. The AI understands multiple languages including
            English, Hindi, Tamil, Telugu, Kannada, and Malayalam.<br><br>
            Start a session from the sidebar to begin.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Message history ──
for msg in st.session_state.messages:
    _render_message(msg)

# ── Chat input ──
if prompt := st.chat_input(
    "Type your message…",
    disabled=(not st.session_state.session_key),
):
    # Validate session
    if not st.session_state.session_key:
        st.error("Please start a session first (sidebar → Start Session).")
        st.stop()

    # Add user message
    user_msg = {
        "role": "user",
        "content": prompt,
        "metadata": {},
        "ts": datetime.now(timezone.utc).strftime("%H:%M"),
    }
    st.session_state.messages.append(user_msg)
    _render_message(user_msg)

    # ── Call backend ──
    typing_placeholder = st.empty()
    with typing_placeholder:
        _render_typing()

    t_start = time.perf_counter()
    raw = _send_message(st.session_state.session_key, prompt)
    latency = round(time.perf_counter() - t_start, 2)
    typing_placeholder.empty()

    # ── Handle errors ──
    if raw is None:
        st.error("No response received. Check your backend configuration.")
        st.stop()

    if "_error" in raw:
        err_type = raw["_error"]
        detail = raw.get("detail", "Unknown error.")

        # Auto-reconnect on session errors
        if err_type in ("connection", "timeout"):
            err_html = (
                f'<div class="zn-bubble zn-bubble-agent" style="color:#f59e0b;">'
                f'⚠️ {detail}</div>'
            )
            st.markdown(f'<div class="zn-msg zn-msg-agent">{err_html}</div>', unsafe_allow_html=True)
        else:
            st.error(f"[{err_type}] {detail}")
        st.stop()

    # ── Parse and display response ──
    bot_text: str = raw.get("response", "").strip() or "*(no response)*"
    scenario: str  = raw.get("detected_scenario", "unknown")
    language: str  = raw.get("detected_language", "english")
    turn: int      = raw.get("turn", st.session_state.turn_count + 1)
    tokens: int    = raw.get("tokens_used", 0)
    cost: float    = raw.get("total_cost", 0.0)
    priority: str  = raw.get("priority_level", "normal")

    # Update session state
    st.session_state.turn_count = turn
    if scenario in CRISIS_SCENARIOS:
        st.session_state.crisis_active = True
    elif scenario in {"positive", "generic", "marks"} and st.session_state.crisis_active:
        # Preserve crisis banner for a few turns; clear only on explicit safe scenarios
        pass  # backend handles crisis lock; we reflect it on next render

    agent_msg = {
        "role": "assistant",
        "content": bot_text,
        "metadata": {
            "detected_scenario": scenario,
            "detected_language": language,
            "turn": turn,
            "priority_level": priority,
            "tokens_used": tokens,
            "total_cost": round(cost, 5),
            "latency_seconds": latency,
            "session_key": st.session_state.session_key,
        },
        "ts": datetime.now(timezone.utc).strftime("%H:%M"),
    }
    st.session_state.messages.append(agent_msg)
    _render_message(agent_msg)

    # Rerun to refresh crisis banner if needed
    if scenario in CRISIS_SCENARIOS and not st.session_state.crisis_active:
        st.rerun()