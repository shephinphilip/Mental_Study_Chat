"""
Zenarck Deep Agent — Production-grade Streamlit frontend
Mental Health Support AI | Behavioral Intelligence Platform
Version: 3.0.0  (no-sidebar layout)
Backend: https://mental-study-chat.onrender.com
"""

import streamlit as st
import requests
import uuid
import time
import os
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be FIRST Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Zenarck Deep Agent",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={"Get Help": None, "Report a bug": None,
                "About": "Zenarck Deep Agent — Mental Health Support AI v3.0"},
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
RENDER_BACKEND = "https://mental-study-chat.onrender.com"

def _get_backend_url() -> str:
    try:
        url = st.secrets["backend"]["url"].strip().rstrip("/")
        if url:
            return url
    except Exception:
        pass
    env = os.environ.get("BACKEND_URL", "").strip().rstrip("/")
    return env if env else RENDER_BACKEND

BACKEND_URL: str = _get_backend_url()
API_BASE:    str = f"{BACKEND_URL}/api/v1"

REQUEST_TIMEOUT = 90    # covers Render cold-start (~30-50s) + LLM call
SESSION_TIMEOUT = 60
HEALTH_TIMEOUT  = 10
MAX_RETRIES     = 2

CRISIS_SCENARIOS = {"crisis"}

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

HELPLINES_HTML = (
    "<strong>Emergency Helplines (India)</strong><br>"
    "AASRA: 91-22-27546669 &nbsp;·&nbsp; "
    "Vandrevala Foundation: 1860-2662-345 &nbsp;·&nbsp; "
    "iCall: 9152987821 &nbsp;·&nbsp; "
    "Emergency: <strong>112</strong>"
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=Fraunces:ital,opsz,wght@0,9..144,300;1,9..144,300&display=swap');

:root {
    --bg:       #080a0f;
    --surf:     #111318;
    --surf2:    #181c26;
    --border:   rgba(255,255,255,0.07);
    --blue:     #3b82f6;
    --violet:   #7c3aed;
    --crisis:   #ef4444;
    --warn:     #f59e0b;
    --ok:       #10b981;
    --text:     #e2e8f4;
    --muted:    #64748b;
    --r:        14px;
    --font:     'Sora', sans-serif;
    --fhead:    'Fraunces', serif;
}

/* ── Hide sidebar toggle arrow & Streamlit header completely ── */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
header { display: none !important; visibility: hidden !important; }

/* ── Body ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
}

/* ── Main column ── */
.block-container {
    max-width: 760px !important;
    padding: 2.5rem 1.5rem 7rem !important;
    margin: 0 auto !important;
}

/* ── Topbar ── */
.zn-topbar {
    display: flex; align-items: center;
    justify-content: space-between;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
    gap: 12px;
}
.zn-brand { display: flex; align-items: center; gap: 12px; }
.zn-logo {
    width: 42px; height: 42px; flex-shrink: 0;
    background: linear-gradient(135deg, var(--blue), var(--violet));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: 0 0 18px rgba(59,130,246,0.35);
}
.zn-name { font-family: var(--fhead); font-size: 1.35rem; color: var(--text); line-height: 1.1; }
.zn-sub  { font-size: 0.70rem; color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; margin-top: 2px; }

/* ── Pills ── */
.zn-pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 11px; border-radius: 20px;
    font-size: 0.73rem; font-weight: 500; white-space: nowrap;
}
.pill-ok    { background: rgba(16,185,129,0.1);  color: var(--ok);    border: 1px solid rgba(16,185,129,0.2); }
.pill-warn  { background: rgba(245,158,11,0.1);  color: var(--warn);  border: 1px solid rgba(245,158,11,0.2); }
.pill-err   { background: rgba(239,68,68,0.1);   color: var(--crisis);border: 1px solid rgba(239,68,68,0.2); }

/* ── Session info strip ── */
.zn-sessbar {
    display: flex; align-items: center;
    background: var(--surf); border: 1px solid var(--border);
    border-radius: var(--r); padding: 9px 16px;
    margin-bottom: 1rem;
    font-size: 0.80rem; color: var(--muted);
}
.zn-sessbar strong { color: var(--text); font-weight: 500; }

/* ── Welcome / info card ── */
.zn-card {
    background: var(--surf); border: 1px solid var(--border);
    border-radius: var(--r); padding: 22px 26px; margin-bottom: 1.25rem;
}
.zn-card-title {
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.07em; color: var(--muted); margin-bottom: 10px;
}
.zn-card p { font-size: 0.93rem; line-height: 1.75; color: var(--text); margin: 0; }

/* ── Warning card ── */
.zn-warn-card {
    background: rgba(245,158,11,0.06); border: 1px solid rgba(245,158,11,0.22);
    border-radius: var(--r); padding: 13px 18px; margin-bottom: 1.25rem;
    font-size: 0.86rem; line-height: 1.65; color: var(--text);
}

/* ── Crisis banner ── */
.zn-crisis {
    background: rgba(239,68,68,0.07); border: 1px solid rgba(239,68,68,0.28);
    border-radius: var(--r); padding: 14px 18px; margin-bottom: 1.25rem;
    font-size: 0.86rem; line-height: 1.75;
}

/* ── Chat bubbles ── */
.zn-msg { margin-bottom: 1.1rem; display: flex; flex-direction: column; }
.zn-msg-user  { align-items: flex-end; }
.zn-msg-agent { align-items: flex-start; }
.zn-bubble {
    max-width: 86%; padding: 13px 17px; border-radius: var(--r);
    font-size: 0.93rem; line-height: 1.68; word-break: break-word;
}
.zn-bubble-user {
    background: linear-gradient(135deg,rgba(59,130,246,0.18),rgba(124,58,237,0.13));
    border: 1px solid rgba(59,130,246,0.22); border-bottom-right-radius: 4px;
}
.zn-bubble-agent {
    background: var(--surf2); border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
}
.zn-bubble-crisis { background: rgba(239,68,68,0.07); border-color: rgba(239,68,68,0.3) !important; }
.zn-meta {
    margin-top: 5px; padding: 0 3px;
    display: flex; align-items: center; gap: 7px;
    font-size: 0.70rem; color: var(--muted);
}
.zn-tag {
    padding: 1px 7px; border-radius: 20px;
    background: rgba(255,255,255,0.04); border: 1px solid var(--border);
    font-size: 0.68rem; font-weight: 500;
}

/* ── Typing dots ── */
.zn-typing { display: flex; align-items: center; gap: 5px; padding: 10px 2px; }
.zn-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--blue); animation: zndot 1.2s ease-in-out infinite; }
.zn-dot:nth-child(2){ animation-delay:.2s; background: var(--violet); }
.zn-dot:nth-child(3){ animation-delay:.4s; }
@keyframes zndot { 0%,100%{ opacity:.25; transform:scale(.75); } 50%{ opacity:1; transform:scale(1.2); } }

/* ── Primary button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--blue), var(--violet)) !important;
    color: #fff !important; border: none !important; border-radius: 9px !important;
    font-family: var(--font) !important; font-weight: 500 !important;
    font-size: 0.88rem !important; padding: 9px 22px !important;
    transition: opacity .18s, transform .1s !important;
}
.stButton > button:hover  { opacity:.84 !important; transform:translateY(-1px) !important; }
.stButton > button:active { transform:translateY(0) !important; }

/* ── Ghost / secondary button wrapper ── */
.ghost-btn .stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-size: 0.80rem !important; padding: 6px 14px !important;
}
.ghost-btn .stButton > button:hover {
    border-color: rgba(255,255,255,0.18) !important;
    color: var(--text) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] { background: var(--surf) !important; border-top: 1px solid var(--border) !important; }
[data-testid="stChatInput"] textarea {
    background: var(--surf2) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 10px !important;
    font-family: var(--font) !important; font-size: 0.93rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] { background: var(--surf) !important; border: 1px solid var(--border) !important; border-radius: var(--r) !important; }
details summary { color: var(--muted) !important; font-size: 0.80rem !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] p { color: var(--muted) !important; font-size: 0.85rem !important; }

/* ── Checkbox ── */
[data-testid="stCheckbox"] label { font-size: 0.83rem !important; color: var(--muted) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.07); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "messages":      [],
    "session_key":   None,
    "user_id":       None,
    "backend_status":"checking",
    "health_ts":     0.0,
    "turn_count":    0,
    "crisis_active": False,
    "show_debug":    False,
    "started_at":    None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.user_id is None:
    st.session_state.user_id = hashlib.sha256(
        str(uuid.uuid4()).encode()).hexdigest()[:12]

# ─────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────
def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    try:
        key = st.secrets["backend"].get("api_key", "")
        if key:
            h["X-API-Key"] = key
    except Exception:
        pass
    return h


def _health() -> str:
    try:
        r = requests.get(f"{BACKEND_URL}/health", headers=_headers(), timeout=HEALTH_TIMEOUT)
        if r.status_code == 200:
            return "ok" if r.json().get("ready", False) else "degraded"
        return "degraded"
    except Exception:
        return "down"


def _start_session(user_id: str) -> Optional[dict]:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(f"{API_BASE}/session/start",
                              json={"user_id": user_id},
                              headers=_headers(), timeout=SESSION_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            return None
        except requests.exceptions.ConnectionError:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(1)
        except Exception:
            return None
    return None


def _chat(session_key: str, message: str) -> Optional[dict]:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(f"{API_BASE}/chat/deep",
                              json={"session_key": session_key, "message": message},
                              headers=_headers(), timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                return {"_error": "rate_limit",
                        "detail": "Server is busy — please wait a moment and try again."}
            return {"_error": "api_error",
                    "detail": f"[{r.status_code}] {r.text[:200]}"}
        except requests.exceptions.Timeout:
            if attempt == MAX_RETRIES - 1:
                return {"_error": "timeout",
                        "detail": "The backend is taking longer than expected. It may be waking up — please try again."}
            time.sleep(1.5)
        except requests.exceptions.ConnectionError:
            return {"_error": "connection",
                    "detail": "Cannot reach the backend. Please try again in a few seconds."}
        except Exception as e:
            return {"_error": "unknown", "detail": str(e)}
    return None

# ─────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────
def _pill(s: str) -> str:
    if s == "ok":
        return '<span class="zn-pill pill-ok">● Online</span>'
    if s == "degraded":
        return '<span class="zn-pill pill-warn">⚠ Degraded</span>'
    if s == "checking":
        return '<span class="zn-pill pill-warn">○ Checking…</span>'
    return '<span class="zn-pill pill-err">✕ Offline</span>'


def _tag(sc: Optional[str]) -> str:
    label = SCENARIO_LABELS.get(sc or "", f"💬 {sc}")
    return f'<span class="zn-tag">{label}</span>'


def _render_msg(msg: dict):
    role = msg["role"]
    content = msg["content"]
    meta = msg.get("metadata", {})
    ts   = msg.get("ts", "")
    sc   = meta.get("detected_scenario", "")
    lat  = meta.get("latency_seconds")

    safe = (content
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("\n", "<br>"))

    if role == "user":
        bubble_cls = "zn-bubble-user"
        row_cls    = "zn-msg-user"
    else:
        extra      = " zn-bubble-crisis" if sc == "crisis" else ""
        bubble_cls = f"zn-bubble-agent{extra}"
        row_cls    = "zn-msg-agent"

    meta_html = ""
    if role == "assistant":
        tag_s = _tag(sc) if sc else ""
        lat_s = f'<span>{lat:.2f}s</span>' if lat else ""
        ts_s  = f'<span>{ts}</span>' if ts else ""
        meta_html = f'<div class="zn-meta">{tag_s}{lat_s}{ts_s}</div>'

    st.markdown(
        f'<div class="zn-msg {row_cls}">'
        f'<div class="zn-bubble {bubble_cls}">{safe}</div>'
        f'{meta_html}</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.show_debug and role == "assistant" and meta:
        with st.expander("🔍 Internal thought process", expanded=False):
            st.json(meta)


def _typing():
    st.markdown(
        '<div class="zn-typing">'
        '<div class="zn-dot"></div><div class="zn-dot"></div><div class="zn-dot"></div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# HEALTH PROBE  (cached 30s)
# ─────────────────────────────────────────────
if time.time() - st.session_state.health_ts > 30:
    st.session_state.backend_status = _health()
    st.session_state.health_ts = time.time()

status = st.session_state.backend_status

# ─────────────────────────────────────────────
# TOPBAR
# ─────────────────────────────────────────────
st.markdown(
    f"""
    <div class="zn-topbar">
        <div class="zn-brand">
            <div class="zn-logo">🧠</div>
            <div>
                <div class="zn-name">Zenarck Deep Agent</div>
                <div class="zn-sub">Mental Health Support AI · Behavioral Intelligence</div>
            </div>
        </div>
        <div>{_pill(status)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# SESSION CONTROLS  (no sidebar — all inline)
# ─────────────────────────────────────────────
if not st.session_state.session_key:
    # ── Welcome card ──
    st.markdown(
        """
        <div class="zn-card">
            <div class="zn-card-title">👋 Welcome</div>
            <p>
            This is a safe, confidential space.<br>
            The AI understands <strong>English, Hindi, Tamil, Telugu, Kannada</strong>
            and <strong>Malayalam</strong>.<br><br>
            Press <strong>Start Session</strong> to begin.
            <br><br>
            <span style="color:#64748b;font-size:0.80rem;">
            ⏳ First connection may take ~30s while the backend wakes up.
            </span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 1.6, 1])
    with c2:
        if st.button("▶  Start Session", type="primary", use_container_width=True):
            with st.spinner("Connecting… (may take up to 30s on cold start)"):
                data = _start_session(st.session_state.user_id)
            if data and data.get("session_key"):
                st.session_state.session_key   = data["session_key"]
                st.session_state.messages      = []
                st.session_state.turn_count    = 0
                st.session_state.crisis_active = False
                st.session_state.started_at    = datetime.now(timezone.utc).strftime("%H:%M UTC")
                st.rerun()
            else:
                st.error(
                    "Could not connect. The backend may still be waking up. "
                    "Wait 30 seconds and try again."
                )

else:
    # ── Active session: compact info strip + ghost action buttons ──
    sk_short = st.session_state.session_key[:22] + "…"

    col_info, col_new, col_clr = st.columns([3.5, 0.9, 0.9])

    with col_info:
        st.markdown(
            f'<div class="zn-sessbar">'
            f'Session active &nbsp;·&nbsp; <strong>{sk_short}</strong>'
            f'&nbsp;·&nbsp; Turn&nbsp;{st.session_state.turn_count}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_new:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("🔄 New", use_container_width=True, key="btn_new"):
            with st.spinner("Starting…"):
                data = _start_session(st.session_state.user_id)
            if data and data.get("session_key"):
                st.session_state.session_key   = data["session_key"]
                st.session_state.messages      = []
                st.session_state.turn_count    = 0
                st.session_state.crisis_active = False
                st.session_state.started_at    = datetime.now(timezone.utc).strftime("%H:%M UTC")
                st.rerun()
            else:
                st.error("Could not start new session.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_clr:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.session_state.messages:
            if st.button("🗑 Clear", use_container_width=True, key="btn_clr"):
                st.session_state.messages      = []
                st.session_state.crisis_active = False
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COLD-START NOTICE  (only while session active + backend offline)
# ─────────────────────────────────────────────
if status == "down" and st.session_state.session_key:
    st.markdown(
        '<div class="zn-warn-card">'
        '⏳ <strong>Backend waking up.</strong> '
        'The server may take 30–50 seconds to respond after a period of inactivity. '
        'Your message will arrive — please be patient.'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# CRISIS BANNER
# ─────────────────────────────────────────────
if st.session_state.crisis_active:
    st.markdown(
        f'<div class="zn-crisis">'
        f'🚨 <strong>If you are in immediate danger, please reach out now.</strong><br><br>'
        f'{HELPLINES_HTML}'
        f'</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# MESSAGE HISTORY
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    _render_msg(msg)

# ─────────────────────────────────────────────
# DEBUG TOGGLE  (shown inline below chat when session active)
# ─────────────────────────────────────────────
if st.session_state.session_key and st.session_state.messages:
    st.session_state.show_debug = st.checkbox(
        "Show internal thought process",
        value=st.session_state.show_debug,
        key="dbg",
    )

# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────
if prompt := st.chat_input(
    "Type your message…",
    disabled=not st.session_state.session_key,
):
    if not st.session_state.session_key:
        st.error("Please start a session first.")
        st.stop()

    user_msg = {
        "role": "user", "content": prompt, "metadata": {},
        "ts": datetime.now(timezone.utc).strftime("%H:%M"),
    }
    st.session_state.messages.append(user_msg)
    _render_msg(user_msg)

    ph = st.empty()
    with ph:
        _typing()

    t0  = time.perf_counter()
    raw = _chat(st.session_state.session_key, prompt)
    lat = round(time.perf_counter() - t0, 2)
    ph.empty()

    if raw is None:
        st.error("No response received.")
        st.stop()

    if "_error" in raw:
        etype  = raw["_error"]
        detail = raw.get("detail", "Unknown error.")
        if etype in ("connection", "timeout"):
            st.markdown(
                f'<div class="zn-msg zn-msg-agent">'
                f'<div class="zn-bubble zn-bubble-agent" style="color:#f59e0b;">'
                f'⚠️ {detail}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.error(detail)
        st.stop()

    bot_text = raw.get("response", "").strip() or "*(no response)*"
    scenario = raw.get("detected_scenario", "unknown")
    language = raw.get("detected_language", "english")
    turn     = raw.get("turn", st.session_state.turn_count + 1)
    tokens   = raw.get("tokens_used", 0)
    cost     = raw.get("total_cost", 0.0)
    priority = raw.get("priority_level", "normal")

    st.session_state.turn_count = turn
    if scenario in CRISIS_SCENARIOS:
        st.session_state.crisis_active = True

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
            "latency_seconds": lat,
        },
        "ts": datetime.now(timezone.utc).strftime("%H:%M"),
    }
    st.session_state.messages.append(agent_msg)
    _render_msg(agent_msg)

    if scenario in CRISIS_SCENARIOS:
        st.rerun()