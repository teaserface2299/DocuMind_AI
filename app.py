import streamlit as st
from rag_pipeline import (
    load_document,
    chunk_text,
    create_vector_store,
    ask_question
)

st.set_page_config(page_title="InsightRAG", layout="wide")

st.title("💬 InsightRAG - AI Knowledge Assistant")
st.write("Upload a TXT or PDF file and chat with it.")

uploaded_file = st.file_uploader(
    "Upload a file",
    type=["txt", "pdf"]
)

if uploaded_file:

    if "index" not in st.session_state:

        with st.spinner("Processing document..."):

            text = load_document(uploaded_file)
            chunks = chunk_text(text)
            index, stored_chunks = create_vector_store(chunks)

            st.session_state.index = index
            st.session_state.chunks = stored_chunks

        st.success("Document processed successfully!")

    question = st.text_input("Ask a question about the document:")

    if question:

        with st.spinner("Generating response..."):

            answer, sources = ask_question(
                question,
                st.session_state.index,
                st.session_state.chunks
            )

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for source in sources:
            st.write(source[:300] + "...")
