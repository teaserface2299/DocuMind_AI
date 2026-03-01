import streamlit as st
import tempfile
import uuid
from datetime import datetime, timedelta
from rag_pipeline import create_qa_system

st.set_page_config(page_title="InsightRAG", layout="wide", page_icon="💬")

# -----------------------------
# 15-MINUTE GLOBAL AUTO-PURGE
# -----------------------------
RESET_INTERVAL = 15 
if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now()

elapsed = datetime.now() - st.session_state.start_time
remaining_seconds = (RESET_INTERVAL * 60) - elapsed.total_seconds()

if remaining_seconds <= 0:
    st.session_state.clear()
    st.warning("🔄 System automatically rebooted after 15 minutes to clear memory.")
    st.rerun()

# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

# -----------------------------
# SIDEBAR: HISTORY & BRANDING
# -----------------------------
with st.sidebar:
    st.title("💬 InsightRAG")
    
    # Global Timer
    mins, secs = divmod(int(remaining_seconds), 60)
    st.error(f"⏳ Auto-Reboot in: {mins:02d}:{secs:02d}")
    
    if st.button("➕ New Chat Session", use_container_width=True):
        st.session_state.current_session_id = None
        st.rerun()

    st.markdown("---")
    st.subheader("Recent Chats")
    
    # List all sessions with a delete button (ChatGPT UI style)
    for sid in list(st.session_state.sessions.keys()):
        cols = st.columns([4, 1])
        # Session Select Button
        if cols[0].button(st.session_state.sessions[sid]["title"], key=f"sel_{sid}", use_container_width=True):
            st.session_state.current_session_id = sid
            st.rerun()
        # Delete Button
        if cols[1].button("🗑️", key=f"del_{sid}"):
            del st.session_state.sessions[sid]
            if st.session_state.current_session_id == sid:
                st.session_state.current_session_id = None
            st.rerun()

    st.markdown("---")
    st.markdown(f"""
    **Created by:** **Yaswanth Dammalapati** AI & ML Engineer | Data Scientist  
    🔗 [LinkedIn](https://www.linkedin.com/in/yaswanth-dammalapati-32091b206/)
    """)

# -----------------------------
# MAIN CONTENT
# -----------------------------
st.title("InsightRAG - Knowledge Assistant")

# CASE 1: No active session selected -> Show File Uploader
if st.session_state.current_session_id is None:
    st.info("👋 Welcome! Please upload a PDF or TXT file to start a new chat session.")
    uploaded_file = st.file_uploader("Upload document", type=["pdf", "txt"])

    if uploaded_file:
        with st.spinner("Processing document..."):
            file_type = uploaded_file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
                tmp.write(uploaded_file.read())
                f_path = tmp.name
            
            # Initialize RAG for this specific file
            qa_func = create_qa_system(f_path, file_type)
            new_id = str(uuid.uuid4())
            
            st.session_state.sessions[new_id] = {
                "title": uploaded_file.name[:20] + "...",
                "qa_system": qa_func,
                "chat_history": [],
                "msg_count": 0
            }
            st.session_state.current_session_id = new_id
            st.rerun()

# CASE 2: Session is active -> Show Chat Interface
else:
    current_sid = st.session_state.current_session_id
    session = st.session_state.sessions[current_sid]
    
    st.caption(f"Active Session: **{session['title']}**")
    
    # Display message history
    for q, a in session["chat_history"]:
        with st.chat_message("user"): st.write(q)
        with st.chat_message("assistant"): st.write(a)

    # Message Limit Logic (6-7 queries)
    if session["msg_count"] < 7:
        user_input = st.chat_input("Ask a question about this document...")
        if user_input:
            with st.chat_message("user"): st.write(user_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    answer, sources = session["qa_system"](user_input, session["chat_history"])
                st.write(answer)
                with st.expander("Reference Sources"):
                    for doc in sources:
                        st.caption(doc.page_content[:250] + "...")
            
            # Update Session History
            session["chat_history"].append((user_input, answer))
            session["msg_count"] += 1
            st.rerun()
    else:
        st.warning("⚠️ Message limit reached (7/7) for this document. Please start a new session.")
