import os
import google.generativeai as genai

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st


# -----------------------------
# Configure Gemini
# -----------------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")


def create_qa_system(file_path, file_type):

    # -----------------------------
    # Load document
    # -----------------------------
    if file_type.lower() == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    documents = loader.load()

    if not documents:
        raise ValueError("No content extracted from file.")

    # -----------------------------
    # Split document
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = splitter.split_documents(documents)
    docs = [doc for doc in docs if doc.page_content.strip()]

    if len(docs) == 0:
        raise ValueError("Document has no valid text content.")

    # -----------------------------
    # Create vector store
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    # -----------------------------
    # Question answering function
    # -----------------------------
    def ask_question(question, chat_history):

        retrieved_docs = vectorstore.similarity_search(question, k=5)

        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs if doc.page_content.strip()]
        )

        history_text = ""
        for q, a in chat_history:
            history_text += f"User: {q}\nAssistant: {a}\n"

        prompt = f"""
You are a helpful AI assistant.

Answer clearly in at least two full paragraphs.
Use ONLY the given context.
If the answer is not found in the context, clearly say that.

Previous Conversation:
{history_text}

Context:
{context}

Question:
{question}
"""

        response = model.generate_content(prompt)

        answer = response.text

        return answer, retrieved_docs

    return ask_question
