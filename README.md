# 🧠 DocuMind AI

**DocuMind AI** is an advanced Retrieval-Augmented Generation (RAG) platform that allows users to have intelligent, context-aware conversations with their documents. Powered by **Google Gemini 2.5 Flash** and **LangChain**, it transforms static PDFs and text files into interactive knowledge bases.

---

## 🚀 Key Features

* **Multi-Session Chat:** ChatGPT-style sidebar to manage multiple document conversations simultaneously.
* **High-Speed RAG:** Uses `FAISS` vector storage and `HuggingFace` embeddings for near-instant retrieval.
* **Smart Memory Purge:** Automatically clears all session data every 15 minutes to ensure privacy and peak performance.
* **Contextual Guardrails:** The AI is strictly limited to answering based on the uploaded document.
* **Source Transparency:** View the exact snippets of text the AI used to generate its answer.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Google Gemini 2.5 Flash
* **Framework:** LangChain
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Document Parsing:** PyPDFLoader & TextLoader

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/documind-ai.git](https://github.com/your-username/documind-ai.git)
cd documind-ai
