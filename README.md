# 🏥 Healthcare RAG Assistant

A **Retrieval-Augmented Generation (RAG) system** built with **Streamlit, FAISS, and OpenAI**, designed to help answer healthcare-related queries from uploaded documents.  
The app supports **multi-document ingestion** (PDF, DOCX, TXT), **Q&A with citations**, **summarization into 5 bullet points**, **glossary generation**, and an **optional reranker** for more accurate answers.  

---

## 🚀 Features
- 📂 Upload multiple **PDF, DOCX, TXT** documents.  
- 🔎 **Semantic search** using FAISS vector store.  
- 💬 **Question answering** with citations from sources.  
- ✍️ **Summarization** into 5 concise bullet points.  
- 📖 **Glossary builder** from domain-specific terms.  
- ⚡ **Optional reranker** for improved relevance.  
- 🎛️ Clean **Streamlit interface** for easy use.  

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) – frontend & UI  
- [LangChain](https://www.langchain.com/) – document loading, chunking, and pipeline  
- [FAISS](https://github.com/facebookresearch/faiss) – vector similarity search  
- [OpenAI GPT](https://platform.openai.com/) – LLM for Q&A, summarization, glossary  

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/healthcare-rag-assistant.git
cd healthcare-rag-assistant

### 2. Install dependencies

pip install -r requirements.txt

### 3. Set your OpenAI API key

OPENAI_API_KEY = "your-openai-api-key"
OPENAI_MODEL = "gpt-4o-mini"

### 4. Run locally

streamlit run app.py



