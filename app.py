import os
import tempfile
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import chromadb
from chromadb.config import Settings

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

# --- File upload
uploaded_files = st.file_uploader(
    "1) Upload documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
    help="Drop multiple files",
)

# --- Save uploads
def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name

doc_paths = [save_uploaded_file(f) for f in uploaded_files] if uploaded_files else []

# --- LLM + Embeddings
llm = ChatOpenAI(model=model_name, api_key=openai_api_key, temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)

# --- Vectorstore
vs = None
if st.button("Ingest"):
    if doc_paths:
        with st.spinner("Building vectorstore..."):
            vs = Chroma.from_documents(
    [],
    embeddings,
    persist_directory="./chroma_store",
    client_settings = Settings(
    persist_directory="chroma_db"
)

          )
            
        st.success("Index updated successfully!")
    else:
        st.warning("Upload at least one document before ingestion.")

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

# --- Tabs (always visible)
st.divider()
st.markdown ("Choose a mode and ask")

tabs = st.tabs(["Q&A", "Summarize", "Glossary"])
ctx_fn = make_context_fn(vs, k=top_k, use_reranker=use_reranker) if vs else None

# --- Q&A tab
with tabs[0]:
    q = st.text_input("Your medical question (e.g., How is measles transmitted?)", key="qa")
    if st.button("Answer", key="qa_btn"):
        if vs is None:
            st.warning("⚠️ Please ingest documents first.")
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
            st.warning("⚠️ Please ingest documents first.")
        elif topic:
            with st.spinner("Summarizing..."):
                chain = ({"context": ctx_fn, "question": RunnablePassthrough()} | sum_prompt | llm | parser)
                summary = chain.invoke(topic)
            st.markdown("#### Summary (5 bullets)")
            st.write(summary)

# --- Glossary tab
with tabs[2]:
    t = st.text_input("Glossary focus (e.g., terms related to tuberculosis)", key="gloss")
    if st.button("Generate Glossary", key="gloss_btn"):
        if vs is None:
            st.warning("⚠️ Please ingest documents first.")
        else:
            query = t if t else "key medical terms"
            with st.spinner("Extracting terms..."):
                chain = ({"context": ctx_fn, "question": RunnablePassthrough()} | terms_prompt | llm | parser)
                gloss = chain.invoke(query)
            st.markdown("#### Medical Glossary (plain English)")
            st.write(gloss)
