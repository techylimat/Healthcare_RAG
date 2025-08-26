import os
import tempfile
import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever  # ✅ BM25 only

# --- Config
st.set_page_config(page_title="🩺 Exceptional Healthcare RAG", layout="wide")
st.title("🩺 Healthcare RAG Assistant")
st.caption("Upload medical docs (PDF/DOCX/TXT), then ask questions, summarize, or get a glossary. Not medical advice.")

# --- Sidebar
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox("Choose model", ["google/flan-t5-small", "google/flan-t5-base"])
top_k = st.sidebar.slider("Number of chunks to retrieve", 2, 8, 4)

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

# --- HuggingFace LLM
hf_token = st.secrets.get("HF_TOKEN")
if not hf_token:
    st.warning("⚠️ Please add your HuggingFace API token in Streamlit secrets.")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

llm = HuggingFaceEndpoint(
    repo_id=model_name,
    huggingfacehub_api_token=hf_token,
    max_new_tokens=512
)

# --- Build BM25 retriever
bm25 = None
if docs:
    with st.spinner("Building BM25 retriever..."):
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = top_k
    st.success("✅ BM25 retriever ready!")

# --- Context retriever
def make_context_fn(bm25):
    return bm25 if bm25 else None

ctx_fn = make_context_fn(bm25)

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

# --- Q&A tab
with tabs[0]:
    q = st.text_input("Your medical question (e.g., How is measles transmitted?)", key="qa")
    if st.button("Answer", key="qa_btn"):
        if bm25 is None:
            st.warning("⚠️ Please upload documents first.")
        elif q:
            with st.spinner("Thinking..."):
                relevant_docs = bm25.get_relevant_documents(q)
                context = "\n".join([doc.page_content for doc in relevant_docs])
                chain = ({"context": context, "question": RunnablePassthrough()} | qa_prompt | llm | parser)
                ans = chain.invoke(q)
            st.markdown("#### Answer")
            st.write(ans)

# --- Summarize tab
with tabs[1]:
    topic = st.text_input("Topic or document theme to summarize (e.g., malaria overview)", key="sum")
    if st.button("Summarize", key="sum_btn"):
        if bm25 is None:
            st.warning("⚠️ Please upload documents first.")
        elif topic:
            with st.spinner("Summarizing..."):
                relevant_docs = bm25.get_relevant_documents(topic)
                context = "\n".join([doc.page_content for doc in relevant_docs])
                chain = ({"context": context, "question": RunnablePassthrough()} | sum_prompt | llm | parser)
                summary = chain.invoke(topic)
            st.markdown("#### Summary (5 bullets)")
            st.write(summary)

# --- Glossary tab
with tabs[2]:
    glossary_topic = st.text_input("Topic to extract glossary from (e.g., diabetes)", key="gloss")
    if st.button("Extract Glossary", key="gloss_btn"):
        if bm25 is None:
            st.warning("⚠️ Please upload documents first.")
        elif glossary_topic:
            with st.spinner("Extracting glossary..."):
                relevant_docs = bm25.get_relevant_documents(glossary_topic)
                context = "\n".join([doc.page_content for doc in relevant_docs])
                chain = ({"context": context, "question": RunnablePassthrough()} | terms_prompt | llm | parser)
                glossary = chain.invoke(glossary_topic)
            st.markdown("#### Glossary")
            st.write(glossary)
