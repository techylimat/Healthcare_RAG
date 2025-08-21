import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import os
import tempfile
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import chromadb
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import time

# exceptions for nicer Streamlit errors
try:
    from openai import RateLimitError, APIConnectionError, APIStatusError
except Exception:
    RateLimitError = Exception
    APIConnectionError = Exception
    APIStatusError = Exception

client = OpenAI()

# --- Config
st.set_page_config(page_title="🩺 Exceptional Healthcare RAG", layout="wide")
st.title("🩺 Healthcare RAG Assitant")
st.caption("Upload medical docs (PDF/DOCX/TXT), then ask questions, summarize, or get a glossary. Not medical advice.")

# --- API key
openai_api_key = st.secrets.get("OPENAI_API_KEY", None)
if not openai_api_key:
    st.warning("Please set your OPENAI_API_KEY in Streamlit secrets to continue.")
    st.stop()

# --- Sidebar
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox("Choose model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
top_k = st.sidebar.slider("Number of chunks to retrieve", 2, 8, 4)
use_reranker = st.sidebar.checkbox("Use reranker for context compression", value=True)

# --- Upload & process documents
uploaded_files = st.file_uploader(
    "Upload PDF or DOCX files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
)

docs = []
if uploaded_files:
    for file in uploaded_files:
        suffix = os.path.splitext(file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file.getbuffer())
            tmp_path = tmp_file.name

        if file.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif file.name.lower().endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = text_splitter.split_documents(docs)

# --- LLM + Embeddings
llm = ChatOpenAI(model=model_name, api_key=openai_api_key, temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# --- Build vectorstore with batching and retry
def embed_in_batches(docs, batch_size=5, retries=3, delay=5):
    all_docs = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        for attempt in range(retries):
            try:
                batch_vs = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory="chroma_db"
                )
                batch_vs.persist()
                all_docs.extend(batch)
                break
            except RateLimitError:
                st.warning(f"Rate limit hit on batch {i // batch_size + 1}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            except (APIConnectionError, APIStatusError) as e:
                st.error(f"OpenAI API error while embedding batch {i // batch_size + 1}: {e}")
                break
            except Exception as e:
                st.error(f"Unexpected error while embedding batch {i // batch_size + 1}: {e}")
                break
    return all_docs

vs = None
if docs:
    with st.spinner("Building vectorstore in batches..."):
        embedded_docs = embed_in_batches(docs)
        if embedded_docs:
            try:
                vs = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
                st.success("Vectorstore built successfully!")
            except Exception as e:
                st.error(f"Error initializing vectorstore: {e}")

# --- Context retriever
def make_context_fn(vs, k, use_reranker):
    if not vs:
        return None
    retriever = vs.as_retriever(search_kwargs={"k": k})
    if use_reranker:
        compressor = LLMChainExtractor.from_llm(llm)
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    return retriever

# --- Prompts
qa_prompt = PromptTemplate.from_template(
    "You are a medical assistant. Using the context below, answer:\n\n{context}\n\nQ: {question}\nA:"
)
sum_prompt = PromptTemplate.from_template(
    "Summarize the following context into 5 bullet points:\n\n{context}\n\nTopic: {question}\nSummary:"
)
terms_prompt = PromptTemplate.from_template(
    "Extract key medical terms and explain simply:\n\n{context}\n\nGlossary:"
)
parser = StrOutputParser()

# --- Tabs
st.divider()
st.markdown("Choose a mode and ask")

tabs = st.tabs(["Q&A", "Summarize", "Glossary"])
ctx_fn = make_context_fn(vs, k=top_k, use_reranker=use_reranker) if vs else None

# --- Q&A tab
with tabs[0]:
    q = st.text_input("Your medical question (e.g., How is measles transmitted?)", key="qa")
    if st.button("Answer", key="qa_btn"):
        if vs is None:
            st.warning("⚠️ Please upload documents first.")
        elif q:
            with st.spinner("Thinking..."):
                chain = ({"context": ctx_fn, "question": RunnablePassthrough()} | qa_prompt | llm | parser)
                ans = chain.invoke(q)
            st.markdown("#### Answer")
            st.write(ans)

# --- Summarize tab
with tabs[1]:
    topic = st.text_input("Topic or document theme to summarize (e.g., malaria overview)", key="sum")
    if st.button("Summarize", key="sum_btn"):
        if vs is None:
            st.warning("⚠️ Please upload documents first.")
        elif topic:
            with st.spinner("Summarizing..."):
                chain = ({"context": ctx_fn, "question": RunnablePassthrough()} | sum_prompt | llm | parser)
                summary = chain.invoke(topic)
            st.markdown("#### Summary (5 bullets)")
            st.write(summary)

# --- Glossary tab
with tabs[2]:
    glossary_topic = st.text_input("Topic to extract glossary from (e.g., diabetes)", key="gloss")
    if st.button("Extract Glossary", key="gloss_btn"):
        if vs is None:
            st.warning("⚠️ Please upload documents first.")
        elif glossary_topic:
            with st.spinner("Extracting glossary..."):
                chain = ({"context": ctx_fn, "question": RunnablePassthrough()} | terms_prompt | llm | parser)
                glossary = chain.invoke(glossary_topic)
            st.markdown("#### Glossary")
            st.write(glossary)
