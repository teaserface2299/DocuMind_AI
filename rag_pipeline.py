import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_qa_system(file_path):

    # =============================
    # 1️⃣ Load Document
    # =============================
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # =============================
    # 2️⃣ Split Text Properly
    # =============================
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = splitter.split_documents(documents)

    # =============================
    # 3️⃣ Create Embeddings
    # =============================
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    # =============================
    # 4️⃣ Load TinyLlama Model
    # =============================
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # Move model to correct device manually
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500,
        temperature=0.7,
        do_sample=True,
    )

    # =============================
    # 5️⃣ Question Function
    # =============================
    def ask_question(question):

        # Retrieve relevant document chunks
        retrieved_docs = vectorstore.similarity_search(question, k=5)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Chat-style prompt format for TinyLlama
        prompt = f"""
<|system|>
You are a helpful AI assistant. Answer clearly and in multiple paragraphs using the provided context.
<|user|>
Context:
{context}

Question:
{question}
<|assistant|>
"""

        result = pipe(prompt)

        full_output = result[0]["generated_text"]

        # Remove prompt part from output
        answer = full_output.split("<|assistant|>")[-1].strip()

        return answer

    return ask_question