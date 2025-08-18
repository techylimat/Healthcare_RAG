# Healthcare_RAG
# 🩺 Healthcare RAG — Streamlit Cloud

**Features**: multi‑doc upload (PDF/DOCX/TXT), FAISS embeddings, Q&A with citations, 5‑bullet summarization, medical glossary, optional reranker.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (must contain `app.py` and `requirements.txt`).
2. Go to https://streamlit.io/cloud → **New app** → pick this repo/branch/file.
3. In your app page → **Settings → Secrets** → add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENAI_MODEL = "gpt-4o-mini"  # optional override
