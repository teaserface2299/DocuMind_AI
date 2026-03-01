import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

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

    # Embeddings (lightweight)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    # 🔥 Load TinyLlama in low-memory mode
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,   # reduce memory
        low_cpu_mem_usage=True
    )

    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # force CPU
        max_new_tokens=250,
        temperature=0.7,
        do_sample=True,
    )

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

Answer clearly in at least two full paragraphs.
Use ONLY the provided context.
If answer is not found, say clearly.

Previous Conversation:
{history_text}

Context:
{context}

Question:
{question}

Answer:
"""

        with torch.no_grad():
            result = pipe(prompt)

        full_output = result[0]["generated_text"]
        answer = full_output.replace(prompt, "").strip()

        return answer, retrieved_docs

    return ask_question
