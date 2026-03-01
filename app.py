import streamlit as st
import tempfile
import uuid
import time
from datetime import datetime, timedelta
from rag_pipeline import create_qa_system

st.set_page_config(page_title="InsightRAG", layout="wide")

# -----------------------------
# 15-MINUTE AUTO-REBOOT LOGIC
# -----------------------------
RESET_INTERVAL = 15  # Minutes
if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now()

# Calculate time remaining
elapsed_time = datetime.now() - st.session_state.start_time
remaining_time = timedelta(minutes=RESET_INTERVAL) - elapsed_time

# If time is up, clear everything
if remaining_time.total_seconds() <= 0:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.warning("Session expired (15 min limit). App rebooted to clear memory.")
    st.rerun()

# -----------------------------
# SIDEBAR & BRANDING
# -----------------------------
st.sidebar.title("💬 InsightRAG")

# Show Countdown Timer
mins, secs = divmod(int(remaining_time.total_seconds()), 60)
st.sidebar.error(f"⏳ Next Auto-Purge in: {mins:02d}:{secs:02d}")
st.sidebar.caption("Data is cleared every 15 mins for performance.")

if st.sidebar.button("➕ New Session"):
    st.session_state.current_session = None

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Created by:** **Yaswanth Dammalapati** AI & ML Engineer | Data Scientist  
🔗 [LinkedIn](https://www.linkedin.com/in/yaswanth-dammalapati-32091b206/)
""")

# Initialize sessions
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "current_session" not in st.session_state:
    st.session_state.current_session = None

# Main page
st.title("InsightRAG - AI Knowledge Assistant")
st.markdown("Upload a **TXT or PDF** file and chat with it.")

uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"])

# Active session logic
if st.session_state.current_session:
    session = st.session_state.sessions[st.session_state.current_session]

    for q, a in session["chat_history"]:
        with st.chat_message("user"): st.markdown(q)
        with st.chat_message("assistant"): st.markdown(a)

    if session["question_count"] < 7:
        question = st.chat_input("Ask something about your document...")
        if question:
            with st.chat_message("user"): st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, sources = session["qa_system"](question, session["chat_history"])
                st.markdown(answer)
                st.markdown("### Sources")
                for doc in sources:
                    st.info(doc.page_content[:300] + "...")

            session["chat_history"].append((question, answer))
            session["question_count"] += 1
    else:
        st.warning("Maximum 7 questions reached for this session.")

# Create new session
if uploaded_file and not st.session_state.current_session:
    file_type = uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    with st.spinner("Processing document..."):
        qa_system = create_qa_system(file_path, file_type)

    session_id = str(uuid.uuid4())
    st.session_state.sessions[session_id] = {
        "title": uploaded_file.name[:25],
        "qa_system": qa_system,
        "chat_history": [],
        "question_count": 0
    }
    st.session_state.current_session = session_id
    st.rerun()
