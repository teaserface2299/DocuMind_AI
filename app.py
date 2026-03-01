import streamlit as st
import tempfile
import uuid
from datetime import datetime, timedelta
from rag_pipeline import create_qa_system

# -----------------------------
# APP CONFIGURATION
# -----------------------------
st.set_page_config(page_title="DocuMind AI", layout="wide", page_icon="🧠")

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
    st.warning("🔄 Security Purge: App data cleared after 15 minutes of inactivity.")
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
    st.title("🧠 DocuMind AI")
    
    # Global Timer Display
    mins, secs = divmod(int(remaining_seconds), 60)
    st.error(f"⏳ Memory Purge In: {mins:02d}:{secs:02d}")
    
    if st.button("➕ New Chat Session", use_container_width=True):
        st.session_state.current_session_id = None
        st.rerun()

    st.markdown("---")
    st.subheader("Chat History")
    
    # List all sessions with a delete button
    for sid in list(st.session_state.sessions.keys()):
        cols = st.columns([4, 1])
        if cols[0].button(st.session_state.sessions[sid]["title"], key=f"sel_{sid}", use_container_width=True):
            st.session_state.current_session_id = sid
            st.rerun()
        if cols[1].button("🗑️", key=f"del_{sid}"):
            del st.session_state.sessions[sid]
            if st.session_state.current_session_id == sid:
                st.session_state.current_session_id = None
            st.rerun()

    st.markdown("---")
    st.markdown(f"""
    **Developed by:** **Yaswanth Dammalapati** AI & ML Engineer | Data Scientist  
    🔗 [LinkedIn Profile](https://www.linkedin.com/in/yaswanth-dammalapati-32091b206/)
    """)

# -----------------------------
# MAIN CONTENT
# -----------------------------
st.title("DocuMind AI — Intelligent Document Collaborator")

if st.session_state.current_session_id is None:
    # --- TEXTUAL DATA SUGGESTION ---
    st.info("👋 **Welcome!** To get started, upload a file below.")
    
    with st.expander("📝 **Important: Supported File Types**", expanded=True):
        st.markdown("""
        * **Standard PDFs:** Files with selectable/searchable text work best.
        * **TXT Files:** Plain text files are fully supported.
        * ⚠️ **Note:** Scanned documents or image-only PDFs (without OCR) cannot be read by this version.
        """)

    uploaded_file = st.file_uploader("Upload document (Text-based PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file:
        with st.spinner("Indexing textual data for AI..."):
            file_type = uploaded_file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
                tmp.write(uploaded_file.read())
                f_path = tmp.name
            
            try:
                qa_func = create_qa_system(f_path, file_type)
                new_id = str(uuid.uuid4())
                
                st.session_state.sessions[new_id] = {
                    "title": uploaded_file.name[:22],
                    "qa_system": qa_func,
                    "chat_history": [],
                    "msg_count": 0
                }
                st.session_state.current_session_id = new_id
                st.rerun()
            except Exception as e:
                st.error(f"Error processing file. Please ensure it contains readable text. (Error: {e})")

# CASE 2: Session is active -> Show Chat Interface
else:
    current_sid = st.session_state.current_session_id
    session = st.session_state.sessions[current_sid]
    
    st.caption(f"Contextual Chatting with: **{session['title']}**")
    
    for q, a in session["chat_history"]:
        with st.chat_message("user"): st.write(q)
        with st.chat_message("assistant"): st.write(a)

    # Message Limit Logic (7 queries)
    if session["msg_count"] < 7:
        user_input = st.chat_input("Ask a question about this document...")
        if user_input:
            with st.chat_message("user"): st.write(user_input)
            with st.chat_message("assistant"):
                with st.spinner("DocuMind is analyzing..."):
                    answer, sources = session["qa_system"](user_input, session["chat_history"])
                st.write(answer)
                with st.expander("Reference Sources"):
                    for doc in sources:
                        st.caption(doc.page_content[:250] + "...")
            
            session["chat_history"].append((user_input, answer))
            session["msg_count"] += 1
            st.rerun()
    else:
        st.warning("⚠️ **Session limit reached (7/7).** To keep the app fast for everyone, please start a new session or delete this one.")
