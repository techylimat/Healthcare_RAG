import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import os
import tempfile
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

# --- Config
st.set_page_config(page_title="🩺 Exceptional Healthcare RAG", layout="wide")
st.title("🩺 Healthcare RAG Assistant")
st.caption("Upload medical docs (PDF/DOCX/TXT), then ask questions, summarize, or get a glossary. Not medical advice.")

# --- Sidebar
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox("Choose model", ["google/flan-t5-small", "google/flan-t5-base"])
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

# --- HuggingFace LLM + Embeddings
hf_token = st.secrets.get("HF_TOKEN", None)
if not hf_token:
    st.warning("⚠️ Please add your HuggingFace API token in Streamlit secrets.")
    st.stop()

llm = HuggingFaceHub(
    repo_id=model_name,
    huggingfacehub_api_token=hf_token,
    model_kwargs={"max_length": 512}
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Build vectorstore
vs = None
if docs:
    with st.spinner("Building vectorstore..."):
        vs = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="chroma_db"
        )
        vs.persist()
    st.success("Vectorstore built successfully!")

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
