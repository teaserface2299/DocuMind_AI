import streamlit as st
import tempfile
from rag_pipeline import create_qa_system

st.set_page_config(page_title="InsightRAG", layout="wide")

st.title("InsightRAG - AI Knowledge Assistant")
st.markdown("Upload a **TXT or PDF** file and ask questions about it.")

# =============================
# Sidebar: JSON Summary
# =============================
st.sidebar.title("Session Summary (JSON)")

if "session_summary" not in st.session_state:
    st.session_state.session_summary = []

if st.session_state.session_summary:
    st.sidebar.json(st.session_state.session_summary)
else:
    st.sidebar.info("No questions asked yet.")

# =============================
# File Upload
# =============================
uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"])

if uploaded_file is not None:

    file_type = uploaded_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("File uploaded successfully!")

    if "qa_system" not in st.session_state:
        with st.spinner("Processing document..."):
            st.session_state.qa_system = create_qa_system(file_path, file_type)
            st.session_state.chat_history = []
            st.session_state.question_count = 0

    # =============================
    # Limit to 7 Questions
    # =============================
    if st.session_state.question_count < 7:

        question = st.text_input("Ask a question about your document:")

        if question:
            with st.spinner("Generating answer..."):
                answer, sources = st.session_state.qa_system(
                    question,
                    st.session_state.chat_history
                )

            st.session_state.chat_history.append((question, answer))
            st.session_state.question_count += 1

            st.subheader("Answer:")
            st.write(answer)

            # Create JSON summary entry
            short_summary = answer[:200] + "..." if len(answer) > 200 else answer

            summary_entry = {
                "question_number": st.session_state.question_count,
                "question": question,
                "short_summary": short_summary
            }

            st.session_state.session_summary.append(summary_entry)

            # Show Sources
            st.subheader("Sources:")
            for i, doc in enumerate(sources):
                st.info(f"Source {i+1}: {doc.page_content[:300]}...")

    else:
        st.warning("Maximum of 7 questions reached for this session.")

# =============================
# Footer
# =============================
st.markdown("---")
st.markdown(
    """
**Created by:** Yaswanth Dammalapati  
AI & ML Engineer | Data Scientist  
🔗 LinkedIn: https://www.linkedin.com/in/yaswanth-dammalapati-32091b206/
"""
)