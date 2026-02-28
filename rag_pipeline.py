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

    # Create vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    # Load TinyLlama
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=500,
        temperature=0.7,
        do_sample=True,
    )

    # Question answering function
    def ask_question(question, chat_history):

        retrieved_docs = vectorstore.similarity_search(question, k=5)

        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs if doc.page_content.strip()]
        )

        history_text = ""
        for q, a in chat_history:
            history_text += f"User: {q}\nAssistant: {a}\n"

        prompt = f"""
<|system|>
You are a helpful AI assistant.
Answer clearly in at least two full paragraphs.
Use only the given context.
If answer not found, say clearly.
<|user|>
Previous Conversation:
{history_text}

Context:
{context}

Question:
{question}
<|assistant|>
"""

        result = pipe(prompt)
        full_output = result[0]["generated_text"]

        if "<|assistant|>" in full_output:
            answer = full_output.split("<|assistant|>")[-1].strip()
        else:
            answer = full_output.strip()

        return answer, retrieved_docs

    return ask_question