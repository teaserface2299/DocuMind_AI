import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import streamlit as st

# --------------------------
# HuggingFace Setup
# --------------------------

HF_TOKEN = st.secrets["HF_TOKEN"]

# ✅ UPDATED ENDPOINT (MANDATORY)
API_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-large"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# --------------------------
# Embedding Model
# --------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------
# Document Loader
# --------------------------

def load_document(file):

    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    else:
        return ""

# --------------------------
# Text Chunking
# --------------------------

def chunk_text(text, chunk_size=500):

    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

# --------------------------
# Vector Store Creation
# --------------------------

def create_vector_store(chunks):

    embeddings = embedding_model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, chunks

# --------------------------
# LLM Query
# --------------------------

def query_llm(prompt):

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return f"Model error: {response.text}"

    result = response.json()

    # Handle HF error response
    if isinstance(result, dict) and "error" in result:
        return f"Model error: {result['error']}"

    if isinstance(result, list):
        return result[0].get("generated_text", "No response generated.")

    return str(result)

# --------------------------
# Main QA Function
# --------------------------

def ask_question(question, index, chunks):

    question_embedding = embedding_model.encode([question])

    D, I = index.search(np.array(question_embedding), k=3)

    retrieved_chunks = [chunks[i] for i in I[0]]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    answer = query_llm(prompt)

    return answer, retrieved_chunks
