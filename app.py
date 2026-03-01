import streamlit as st
import tempfile
import uuid
from rag_pipeline import create_qa_system

st.set_page_config(page_title="InsightRAG", layout="wide")

# Session storage
if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session" not in st.session_state:
    st.session_state.current_session = None

# Sidebar
st.sidebar.title("💬 InsightRAG")

if st.sidebar.button("➕ New Session"):
    st.session_state.current_session = None

st.sidebar.markdown("---")

for session_id, session_data in list(st.session_state.sessions.items()):
    col1, col2 = st.sidebar.columns([4, 1])

    if col1.button(session_data["title"], key=session_id):
        st.session_state.current_session = session_id

    if col2.button("🗑", key=f"delete_{session_id}"):
        del st.session_state.sessions[session_id]
        if st.session_state.current_session == session_id:
            st.session_state.current_session = None
        st.rerun()

# Main page
st.title("InsightRAG - AI Knowledge Assistant")
st.markdown("Upload a **TXT or PDF** file and chat with it.")

uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"])

# Active session
if st.session_state.current_session:

    session = st.session_state.sessions[st.session_state.current_session]

    for q, a in session["chat_history"]:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    if session["question_count"] < 7:

        question = st.chat_input("Ask something about your document...")

        if question:

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, sources = session["qa_system"](
                        question,
                        session["chat_history"]
                    )
                st.markdown(answer)

                st.markdown("### Sources")
                for doc in sources:
                    st.info(doc.page_content[:300] + "...")

            session["chat_history"].append((question, answer))
            session["question_count"] += 1

    else:
        st.warning("Maximum 7 questions reached for this session.")

else:
    st.info("Create a new session and upload a document to start chatting.")

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
        "question_count": 0,
        "file_name": uploaded_file.name
    }

    st.session_state.current_session = session_id
    st.rerun()

st.markdown("---")
st.markdown("Created by Yaswanth Dammalapati")
