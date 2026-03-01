import os
from huggingface_hub import InferenceClient

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

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    # HuggingFace API client
    client = InferenceClient(
        model="HuggingFaceH4/zephyr-7b-beta",
        token=os.environ["HF_TOKEN"],
    )

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
If the answer is not found, say:
"I could not find the answer in the document."

Previous Conversation:
{history_text}

Context:
{context}

Question:
{question}

Answer:
"""

        response = client.text_generation(
            prompt,
            max_new_tokens=300,
            temperature=0.7,
        )

        answer = response.strip()

        return answer, retrieved_docs

    return ask_question
