import streamlit as st
import tempfile
from rag_pipeline import create_qa_system

st.set_page_config(page_title="InsightRAG", layout="wide")

st.title("InsightRAG - AI Knowledge Assistant")
st.write("Upload a .txt file and ask questions about it.")

uploaded_file = st.file_uploader("Upload a text file", type="txt")

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("File uploaded successfully!")

    if "qa_system" not in st.session_state:
        with st.spinner("Processing document..."):
            st.session_state.qa_system = create_qa_system(file_path)

    question = st.text_input("Ask a question about your document:")

    if question:
        with st.spinner("Generating answer..."):
            answer = st.session_state.qa_system(question)

        st.subheader("Answer:")
        st.write(answer)