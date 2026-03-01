import os
import requests
import time

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_qa_system(file_path, file_type):

    # Load document
    if file_type.lower() == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    docs = splitter.split_documents(documents)
    docs = [doc for doc in docs if doc.page_content.strip()]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
    headers = {
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}"
    }

    def query_llm(prompt):

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.7,
                "return_full_text": False
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()

        # Handle model loading case
        if isinstance(result, dict) and "estimated_time" in result:
            time.sleep(result["estimated_time"])
            response = requests.post(API_URL, headers=headers, json=payload)
            result = response.json()

        if isinstance(result, list):
            return result[0]["generated_text"]

        return "Model is currently unavailable. Please try again."

    def ask_question(question, chat_history):

        retrieved_docs = vectorstore.similarity_search(question, k=4)

        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        history_text = ""
        for q, a in chat_history:
            history_text += f"User: {q}\nAssistant: {a}\n"

        prompt = f"""
You are a helpful AI assistant.

Answer clearly in at least two paragraphs.
Use ONLY the provided context.
If answer is not found, clearly say so.

Previous Conversation:
{history_text}

Context:
{context}

Question:
{question}

Answer:
"""

        answer = query_llm(prompt)

        return answer.strip(), retrieved_docs

    return ask_question
