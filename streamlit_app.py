import streamlit as st
import requests
import uuid
import json
import time

# ─── CONFIGURATION ───
st.set_page_config(
    page_title="Zenarck Deep Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern aesthetic
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    .stButton button {
        border-radius: 20px;
    }
    .debug-info {
        font-size: 0.8em;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_key" not in st.session_state:
    st.session_state.session_key = None

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:8]

if "api_port" not in st.session_state:
    st.session_state.api_port = "8002"  # Default to the most recent stable port

# ─── SIDEBAR ───
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Port Selection
    st.session_state.api_port = st.text_input("API Port", value=st.session_state.api_port, help="Port where deep_agent_system is running (e.g., 8000, 8001, 8002)")
    base_url = f"http://localhost:{st.session_state.api_port}/api/v1"
    
    st.divider()
    
    st.subheader("Session Management")
    st.caption(f"User ID: `{st.session_state.user_id}`")
    
    if st.button("Start New Session", type="primary"):
        with st.spinner("Connecting to Agent..."):
            try:
                payload = {"user_id": st.session_state.user_id}
                response = requests.post(f"{base_url}/session/start", json=payload, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.session_key = data.get("session_key")
                    st.session_state.messages = [] # Clear history on new session
                    st.success("Session Started!")
                    st.rerun()
                else:
                    st.error(f"Failed: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Connection refused at port {st.session_state.api_port}. Is the server running?")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.session_key:
        st.success(f"Active Session:\n`{st.session_state.session_key}`")
    else:
        st.warning("No active session. Click 'Start New Session'.")

    st.divider()
    st.subheader("Debug Info")
    show_debug = st.checkbox("Show Scenario Details", value=True)

# ─── MAIN CHAT INTERFACE ───
st.title("🧠 Zenarck Deep Agent")
st.markdown("Mental Health Support AI System | Behavioral Intelligence")

# Display Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if show_debug and "metadata" in msg:
            with st.expander("Internal Thought Process", expanded=False):
                st.json(msg["metadata"])

# Input Handling
if prompt := st.chat_input("Type your message here..."):
    if not st.session_state.session_key:
        st.error("Please start a session first!")
    else:
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("typing...")
            
            try:
                start_time = time.time()
                payload = {
                    "session_key": st.session_state.session_key,
                    "message": prompt
                }
                
                response = requests.post(f"{base_url}/chat/deep", json=payload, timeout=30)
                latency = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    bot_response = data.get("response", "No response content.")
                    
                    # Metadata for debug
                    metadata = {
                        "detected_scenario": data.get("detected_scenario"),
                        "detected_language": data.get("detected_language"),
                        "latency_seconds": round(latency, 2)
                    }
                    
                    # Update placeholder
                    message_placeholder.write(bot_response)
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": bot_response,
                        "metadata": metadata
                    })
                    
                    if show_debug:
                        with st.expander("Internal Thought Process", expanded=True):
                            st.json(metadata)
                            
                else:
                    message_placeholder.error(f"API Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                message_placeholder.error(f"Could not connect to server at port {st.session_state.api_port}")
            except Exception as e:
                message_placeholder.error(f"Error: {e}")
