import os
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Optional reranker
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_OK = True
except Exception:
    CROSS_ENCODER_OK = False

# ------------ Config ------------
st.set_page_config(page_title="Healthcare RAG", page_icon="🩺", layout="wide")
load_dotenv()

# Secrets/keys
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not set. In Streamlit Cloud, go to App → Settings → Secrets and add it.")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Local folders (persist for current app instance; cleared on redeploy)
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
STORE_DIR = Path("store"); STORE_DIR.mkdir(exist_ok=True)
CHROMA_PATH = str(STORE_DIR / "chroma_db")

# Components
embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)

# ------------ Loaders ------------

def load_docs(files: List[st.runtime.uploaded_file_manager.UploadedFile]):
    all_docs = []
    for f in files:
        path = DATA_DIR / f.name
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        ext = path.suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(str(path))
        elif ext == ".docx":
            loader = Docx2txtLoader(str(path))
        elif ext in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            st.warning(f"Unsupported file type: {f.name}"); continue
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = f.name
        all_docs.extend(docs)
    return all_docs

# ------------ Chunking ------------

def chunk_docs(docs, size=700, overlap=120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n","\n"," ", ""]
    )
    return splitter.split_documents(docs)

# ------------ Vector store (Chroma) ------------

def build_or_update_index(chunks):
    if os.path.exists(CHROMA_PATH):
        vs = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        vs.add_documents(chunks)
    else:
        vs = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    vs.persist()
    return vs

@st.cache_resource(show_spinner=False)
def load_index():
    if os.path.exists(CHROMA_PATH):
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return None

# ------------ Retrieval helpers ------------

def format_docs(docs):
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", d.metadata.get("source_file", "unknown"))
        page = d.metadata.get("page")
        tag = f"[Source {i} — {src}"
        if page is not None:
            tag += f", p.{page+1}"
        tag += "]"
        blocks.append(f"{tag}\n{d.page_content}")
    return "\n\n".join(blocks)

@st.cache_resource(show_spinner=False)
def get_reranker():
    if not CROSS_ENCODER_OK:
        return None
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ------------ Prompts ------------
qa_prompt = ChatPromptTemplate.from_template(
    """
You are a careful medical assistant.
Answer **only** using the context. If unsure, say you don't know based on these documents.
Write concise, structured answers. Cite evidence inline as [Source N].

Context:\n{context}

Question: {question}
Answer:
"""
)

sum_prompt = ChatPromptTemplate.from_template(
    """
You are a summarizer. Using only the context, produce **exactly 5 crisp bullet points** covering definition, key symptoms/signs, transmission (if applicable), prevention/treatment, and key risks. Cite sources inline as [Source N].

Context:\n{context}

Topic: {question}
Summary:
"""
)

terms_prompt = ChatPromptTemplate.from_template(
    """
You are a medical glossary generator. From the context, extract **up to 8 important medical terms** and explain each in simple language for laypersons. Use the format:
- Term: plain-English explanation (one sentence)
Only use information inferable from the context; do not invent.

Context:\n{context}

Glossary:
"""
)

parser = StrOutputParser()

# ------------ Chains ------------

def make_context_fn(vs, k, use_reranker):
    retriever = vs.as_retriever(search_kwargs={"k": k})
    reranker = get_reranker() if use_reranker else None

    def _ctx(query: str):
        docs = retriever.get_relevant_documents(query)
        if reranker is not None:
            pairs = [[query, d.page_content] for d in docs]
            scores = reranker.predict(pairs)
            docs = [d for d, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:k]]
        return format_docs(docs)
    return _ctx

# ------------ UI ------------
st.title("🩺 Exceptional Healthcare RAG — Streamlit Cloud")
st.caption("Upload medical docs (PDF/DOCX/TXT), then ask questions, summarize, or get a glossary. Not medical advice.")

with st.sidebar:
    st.subheader("Settings")
    top_k = st.slider("Retriever k", 2, 12, 5)
    use_reranker = st.checkbox("Use reranker (better relevance)", value=False, disabled=not CROSS_ENCODER_OK)
    st.write(":grey[Embeddings:] ")
    st.code(EMB_MODEL, language="")
    st.write(":grey[LLM:]")
    st.code(OPENAI_MODEL, language="")

st.markdown("### 1) Upload documents (PDF, DOCX, TXT)")
files = st.file_uploader("Drop multiple files", type=["pdf","docx","txt","md"], accept_multiple_files=True)

c1, c2, c3 = st.columns(3)
with c1:
    ingest_clicked = st.button("Ingest / Update Index", type="primary")
with c2:
    clear_clicked = st.button("Clear Index")
with c3:
    reload_clicked = st.button("Reload Saved Index")

vs = None

if ingest_clicked:
    if not files:
        st.warning("Please upload at least one file.")
    else:
        with st.spinner("Loading & chunking documents..."):
            docs = load_docs(files)
            chunks = chunk_docs(docs)
        with st.spinner("Building Chroma index (persisting)..."):
            vs = build_or_update_index(chunks)
        st.success("Index ready & saved.")

if clear_clicked:
    if os.path.exists(CHROMA_PATH):
        import shutil; shutil.rmtree(CHROMA_PATH)
        st.success("Cleared Chroma index.")
    else:
        st.info("No index found to clear.")

if reload_clicked:
    vs = load_index()
    if vs is None:
        st.info("No saved index found yet. Please ingest documents.")
    else:
        st.success("Index loaded from disk.")

# Auto-load existing index on first run
if vs is None:
    vs = load_index()

st.divider()

if vs is None:
    st.info("Upload and ingest documents to begin.")
else:
    st.markdown("### 2) Choose a mode and ask")
    tabs = st.tabs(["Q&A", "Summarize", "Glossary"])
    ctx_fn = make_context_fn(vs, k=top_k, use_reranker=use_reranker)

    with tabs[0]:
        q = st.text_input("Your medical question (e.g., How is measles transmitted?)", key="qa")
        if st.button("Answer", key="qa_btn") and q:
            with st.spinner("Thinking..."):
                chain = ({"context": ctx_fn, "question": RunnablePassthrough()} | qa_prompt | llm | parser)
                ans = chain.invoke(q)
            st.markdown("#### Answer")
            st.write(ans)

    with tabs[1]:
        topic = st.text_input("Topic or document theme to summarize (e.g., malaria overview)", key="sum")
        if st.button("Summarize", key="sum_btn") and topic:
            with st.spinner("Summarizing..."):
                chain = ({"context": ctx_fn, "question": RunnablePassthrough()} | sum_prompt | llm | parser)
                summary = chain.invoke(topic)
            st.markdown("#### Summary (5 bullets)")
            st.write(summary)

    with tabs[2]:
        t = st.text_input("Glossary focus (e.g., terms related to tuberculosis)", key="gloss")
        if st.button("Generate Glossary", key="gloss_btn"):
            query = t if t else "key medical terms"
            with st.spinner("Extracting terms..."):
                chain = ({"context": ctx_fn, "question": RunnablePassthrough()} | terms_prompt | llm | parser)
                gloss = chain.invoke(query)
            st.markdown("#### Medical Glossary (plain English)")
            st.write(gloss)

st.divider()
st.markdown("**Note:** Results may omit details or drift as documents update. Verify with primary sources.")
