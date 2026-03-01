# 💬 InsightRAG – AI Knowledge Assistant

InsightRAG is a Retrieval-Augmented Generation (RAG) based AI assistant that allows users to upload PDF or TXT documents and chat with them intelligently.

Built using:
- Streamlit
- LangChain
- FAISS
- HuggingFace Transformers
- TinyLlama (Local LLM)

---

## 🚀 Features

- Upload PDF or TXT documents
- Intelligent context-based question answering
- Retrieval-Augmented Generation (RAG)
- Multi-session chat support
- Source document preview
- Fully local LLM (No OpenAI API required)
- Question limit per session (7 questions)

---

## 🧠 How It Works

1. Document is uploaded
2. Text is split into chunks
3. Embeddings are created using `all-MiniLM-L6-v2`
4. FAISS vector store is built
5. TinyLlama generates answers using retrieved context

---

## 🛠 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/InsightRAG.git
cd InsightRAG