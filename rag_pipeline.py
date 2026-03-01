import os
import requests

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
    if not documents:
        raise ValueError("No content extracted from file.")

    # Split document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = splitter.split_documents(documents)
    docs = [doc for doc in docs if doc.page_content.strip()]

    if len(docs) == 0:
        raise ValueError("Document has no valid text content.")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

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

Use ONLY the context below to answer.
If the answer is not in the context, say:
"I could not find the answer in the provided document."

Previous Conversation:
{history_text}

Context:
{context}

Question:
{question}

Answer:
"""

        API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b-it"

        headers = {
            "Authorization": f"Bearer {os.environ['HF_TOKEN']}"
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.7
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()

        if isinstance(result, list):
            answer = result[0]["generated_text"]
            answer = answer.replace(prompt, "").strip()
        else:
            answer = "Error generating response."

        return answer, retrieved_docs

    return ask_question
