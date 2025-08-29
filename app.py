import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# ===============================
# Streamlit App
# ===============================
st.set_page_config(page_title="Healthcare RAG", page_icon="🩺", layout="wide")
st.title("🩺 Healthcare RAG Assistant")
st.write("Upload healthcare PDFs and ask AI questions about them!")

# ===============================
# Hugging Face Token from Secrets
# ===============================
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HF_TOKEN"]

# ===============================
# File Upload
# ===============================
uploaded_files = st.file_uploader("📂 Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    docs = []
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader(uploaded_file.name)
        docs.extend(loader.load())

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store in Chroma (instead of FAISS)
    db = Chroma.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    # HuggingFace LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.3, "max_length": 512}
    )

    # RetrievalQA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    # User input
    query = st.text_input("💬 Ask a question about your documents:")
    if query:
        with st.spinner("🤔 Thinking..."):
            answer = qa.run(query)
        st.success(answer)
