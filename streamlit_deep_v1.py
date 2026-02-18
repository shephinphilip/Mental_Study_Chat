#!/usr/bin/env python3
"""
Dr. Mind v5.1 — Streamlit Frontend
Calls the FastAPI backend at API_BASE_URL.

Run:
  streamlit run streamlit_app.py
"""

import streamlit as st
import requests
from datetime import datetime

# ─── Config ──────────────────────────────────────────────────────────────────

API_BASE_URL = "http://localhost:8000"   # change if backend is remote

# ─── Page setup ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dr. Mind — Clinical Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1117;
    color: #e0e0e0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #1e2535;
}

/* ── Chat bubbles ── */
.user-bubble {
    background: #1a3a5c;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 6px 0 6px 60px;
    color: #e8f4fd;
    font-size: 0.95rem;
    line-height: 1.55;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
}
.ai-bubble {
    background: #1e2535;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    margin: 6px 60px 6px 0;
    color: #d8dde8;
    font-size: 0.95rem;
    line-height: 1.65;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    border-left: 3px solid #3a7bd5;
}
.ai-bubble.crisis  { border-left-color: #e53935; }
.ai-bubble.violence { border-left-color: #f57c00; }
.ai-bubble.harassment { border-left-color: #8e24aa; }

/* ── Classification badge ── */
.badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    padding: 2px 8px;
    border-radius: 999px;
    margin-top: 5px;
    opacity: 0.75;
    text-transform: uppercase;
}
.badge-SAFE            { background:#1b3a2a; color:#4caf50; }
.badge-TEACHER_SYSTEMIC { background:#1a2a3a; color:#64b5f6; }
.badge-EXAM_STRESS     { background:#2a2215; color:#ffb74d; }
.badge-NEGATIVE        { background:#2a1a2a; color:#ce93d8; }
.badge-FAMILY_CONFLICT { background:#2a1a1a; color:#ef9a9a; }
.badge-RELATIONSHIP    { background:#1a1a2a; color:#90caf9; }
.badge-BODY_IMAGE      { background:#1a2a1a; color:#a5d6a7; }
.badge-OCD             { background:#2a2a1a; color:#fff176; }
.badge-MARKS           { background:#1a2a2a; color:#80cbc4; }
.badge-CRISIS          { background:#3a1a1a; color:#ef5350; font-weight:700; }
.badge-VIOLENCE        { background:#3a1f0a; color:#ffa726; font-weight:700; }
.badge-SUBSTANCE       { background:#1a2a1a; color:#aed581; }
.badge-SEXUAL_HARASSMENT { background:#2a1a2a; color:#f48fb1; font-weight:700; }
.badge-AMBIGUOUS       { background:#222; color:#9e9e9e; }
.badge-MISCHIEVOUS     { background:#222; color:#80cbc4; }

/* ── Risk badge ── */
.risk-low      { color:#4caf50; font-weight:600; }
.risk-moderate { color:#ffb74d; font-weight:600; }
.risk-high     { color:#ef5350; font-weight:600; }
.risk-crisis   { color:#e53935; font-weight:700; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.5;} }

/* ── Input box ── */
[data-testid="stChatInput"] textarea {
    background: #1e2535 !important;
    color: #e0e0e0 !important;
    border: 1px solid #2a3550 !important;
    border-radius: 12px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #1a3a5c;
    color: #90caf9;
    border: 1px solid #2a4a6c;
    border-radius: 8px;
    font-size: 0.85rem;
}
.stButton > button:hover {
    background: #1e4a7a;
    color: #bbdefb;
}

/* ── Usage pill ── */
.usage-pill {
    display: inline-flex;
    gap: 10px;
    align-items: center;
    font-size: 0.68rem;
    color: #5a6a8a;
    margin-top: 6px;
    letter-spacing: 0.02em;
}
.usage-pill span {
    background: #12171f;
    border: 1px solid #1e2a3a;
    border-radius: 6px;
    padding: 1px 7px;
}
.usage-pill .cost {
    color: #81c784;
    border-color: #1e3a2a;
    background: #111e17;
}
.usage-pill .calls {
    color: #64b5f6;
    border-color: #1a2a3a;
}

/* ── Alert box ── */
.crisis-alert {
    background: #3a1a1a;
    border: 1px solid #e53935;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #ffcdd2;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def api_get(path: str, params: dict = None) -> dict | list | None:
    try:
        r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach Dr. Mind API. Is the server running? (`uvicorn api:app --reload`)")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach Dr. Mind API. Is the server running?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def risk_badge(level: str) -> str:
    icons = {"low": "🟢", "moderate": "🟡", "high": "🔴", "crisis": "🆘"}
    return f'<span class="risk-{level}">{icons.get(level, "⚪")} {level.upper()}</span>'


def classification_badge(cls: str) -> str:
    return f'<span class="badge badge-{cls}">{cls.replace("_", " ")}</span>'


def usage_pill(usage: dict | None) -> str:
    """Render a compact token/cost strip below an AI bubble."""
    if not usage:
        return ""
    calls = usage.get("llm_calls", 0)
    total = usage.get("total_tokens", 0)
    inp   = usage.get("prompt_tokens", 0)
    out   = usage.get("completion_tokens", 0)
    cost  = usage.get("cost_usd", 0.0)
    return (
        f'<div class="usage-pill">'
        f'<span class="calls">🔁 {calls} call{"s" if calls != 1 else ""}</span>'
        f'<span>🔤 {total:,} tok &nbsp;({inp:,} in / {out:,} out)</span>'
        f'<span class="cost">💰 ${cost:.5f}</span>'
        f'</div>'
    )


def bubble_class(cls: str) -> str:
    danger = {"CRISIS": "crisis", "VIOLENCE": "violence", "SEXUAL_HARASSMENT": "harassment"}
    return f"ai-bubble {danger.get(cls, '')}"


# ─── Session state initialisation ────────────────────────────────────────────

def init_state():
    defaults = {
        "session_id":    None,
        "student_id":    None,
        "profile":       None,
        "messages":      [],   # [{"role": "user"|"ai", "text": str, "classification": str, "usage": dict|None}]
        "profiles_list": None,
        "last_cls":          "SAFE",
        "detected_language": "ENGLISH",
        # cumulative session usage
        "session_total_tokens": 0,
        "session_total_cost":   0.0,
        "session_total_calls":  0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Dr. Mind")
    st.markdown("*Clinical Psychiatric AI for Indian Teenagers*")
    st.divider()

    # Load profiles on first render
    if st.session_state.profiles_list is None:
        profiles = api_get("/api/profiles")
        st.session_state.profiles_list = profiles or []

    profiles_list = st.session_state.profiles_list

    if profiles_list:
        profile_options = {
            p["student_id"]: f"{p['name']}, {p['age']}y — {p['city']} [{p['risk_level'].upper()}]"
            for p in profiles_list
        }

        selected_id = st.selectbox(
            "Select Patient",
            options=list(profile_options.keys()),
            format_func=lambda x: profile_options[x],
            key="profile_selector",
        )

        selected_meta = next(p for p in profiles_list if p["student_id"] == selected_id)

        st.markdown(f"""
**Chief Concern:**  
{selected_meta['chief_concern']}

**Risk Level:** {risk_badge(selected_meta['risk_level'])}
""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Start Session", use_container_width=True):
                with st.spinner("Initialising agent..."):
                    resp = api_post("/api/session/start", {"student_id": selected_id})
                if resp:
                    st.session_state.session_id = resp["session_id"]
                    st.session_state.student_id = resp["student_id"]
                    st.session_state.profile    = resp
                    st.session_state.messages   = []
                    st.session_state.last_cls          = "SAFE"
                    st.session_state.detected_language = "ENGLISH"
                    st.session_state.session_total_tokens = 0
                    st.session_state.session_total_cost   = 0.0
                    st.session_state.session_total_calls  = 0
                    st.success("Session started")
                    st.rerun()

        with col2:
            if st.button("🔄 Clear Chat", use_container_width=True):
                if st.session_state.session_id:
                    api_post("/api/session/clear", {"session_id": st.session_state.session_id})
                st.session_state.messages = []
                st.rerun()
    else:
        st.warning("No profiles loaded. Is the API running?")

    st.divider()

    # Active session info
    if st.session_state.session_id:
        p = st.session_state.profile
        st.markdown(f"""
**Active Session**  
`{st.session_state.session_id}`  
Patient: **{p['name']}**, {p['age']}y  
Risk: {risk_badge(p['risk_level'])}  
Language: 🌐 **{st.session_state.detected_language}**  
Last classification: {classification_badge(st.session_state.last_cls)}
""", unsafe_allow_html=True)
        # ── Session usage summary ─────────────────────────────────────────
        st.markdown(f"""
<div style="font-size:0.78rem; color:#5a6a8a; margin-top:6px; line-height:1.9;">
📊 <b style="color:#9eaec4">Session Usage</b><br>
&nbsp;🔁 LLM Calls &nbsp; <code>{st.session_state.session_total_calls}</code><br>
&nbsp;🔤 Tokens &nbsp;&nbsp;&nbsp;&nbsp;<code>{st.session_state.session_total_tokens:,}</code><br>
&nbsp;💰 Cost &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<code style="color:#81c784">${st.session_state.session_total_cost:.4f}</code>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("*No active session. Select a patient and click Start Session.*")

    st.divider()

    # History panel
    if st.session_state.student_id:
        if st.button("📋 View DB History", use_container_width=True):
            records = api_get(f"/api/history/{st.session_state.student_id}", params={"limit": 8})
            if records:
                st.markdown("**Recent Encounters**")
                for r in records:
                    ts  = r["timestamp"][:16].replace("T", " ")
                    cls = r.get("classification", "?")
                    msg = (r.get("user_message") or "")[:40]
                    st.markdown(
                        f"`{ts}` {classification_badge(cls)} {msg}…",
                        unsafe_allow_html=True,
                    )


# ─── Main chat area ───────────────────────────────────────────────────────────

st.markdown("## 🧠 Dr. Mind — Clinical Chat")

if not st.session_state.session_id:
    st.markdown("""
<div style="
    text-align:center; padding:60px 20px;
    color:#4a5568; font-size:1.1rem;
    border: 1px dashed #2a3550;
    border-radius:16px; margin-top:40px;
">
    👈 Select a patient profile from the sidebar and click <b>Start Session</b> to begin.
</div>
""", unsafe_allow_html=True)
    st.stop()

# ── Crisis alert banner ────────────────────────────────────────────────────────

if st.session_state.last_cls == "CRISIS":
    st.markdown("""
<div class="crisis-alert">
    🆘 <strong>CRISIS ALERT</strong> — Student may be in immediate danger.
    Emergency resources have been shared in the chat.
    Consider escalating to a human counsellor immediately.
</div>
""", unsafe_allow_html=True)

elif st.session_state.last_cls == "SEXUAL_HARASSMENT":
    st.markdown("""
<div style="background:#2a1a2a; border:1px solid #8e24aa; border-radius:10px;
            padding:12px 18px; margin:8px 0; color:#f3e5f5; font-size:0.9rem;">
    🟣 <strong>SENSITIVE DISCLOSURE</strong> — Handle with care. Avoid re-traumatisation.
</div>
""", unsafe_allow_html=True)

# ── Render chat history ────────────────────────────────────────────────────────

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">🧑 {msg["text"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            cls    = msg.get("classification", "SAFE")
            bubble = bubble_class(cls)
            badge  = classification_badge(cls)
            pill   = usage_pill(msg.get("usage"))
            text   = msg["text"].replace("\n", "<br>")
            st.markdown(
                f'<div class="{bubble}">🧠 {text}<br>{badge}{pill}</div>',
                unsafe_allow_html=True,
            )

# ── Chat input ────────────────────────────────────────────────────────────────

profile_name = st.session_state.profile["name"]

if prompt := st.chat_input(f"Message as {profile_name}…"):
    # Add user bubble immediately
    st.session_state.messages.append({"role": "user", "text": prompt})
    st.rerun()   # show user bubble before waiting for API

# ── Process pending user message (if last message is from user and has no AI reply yet) ──
messages = st.session_state.messages
if messages and messages[-1]["role"] == "user":
    # Check if AI hasn't replied yet (prevent double-send on rerun)
    pending = messages[-1]["text"]

    with st.spinner("Dr. Mind is thinking…"):
        resp = api_post("/api/chat", {
            "session_id": st.session_state.session_id,
            "message":    pending,
        })

    if resp:
        usage = resp.get("usage", {})
        st.session_state.last_cls          = resp.get("classification", "SAFE")
        st.session_state.detected_language = resp.get("detected_language", "ENGLISH")
        # Accumulate session-level usage
        st.session_state.session_total_tokens += usage.get("total_tokens", 0)
        st.session_state.session_total_cost   += usage.get("cost_usd", 0.0)
        st.session_state.session_total_calls  += usage.get("llm_calls", 0)
        st.session_state.messages.append({
            "role":           "ai",
            "text":           resp["response"],
            "classification": resp["classification"],
            "inquiry_stage":  resp.get("inquiry_stage", "initial"),
            "usage":          usage,
        })
        st.rerun()

# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; color:#3a4a5a; font-size:0.75rem; margin-top:40px; padding:10px 0;">
    Dr. Mind v5.1 — For clinical research purposes only. Not a substitute for professional care.
</div>
""", unsafe_allow_html=True)