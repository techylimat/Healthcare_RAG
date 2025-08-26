import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_stuff_documents_chain
from langchain.chains.combine_documents import create_retrieval_chain
from langchain_community.vectorstores import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Healthcare RAG", page_icon="🩺", layout="wide")
st.title("🩺 Healthcare RAG Chatbot")
st.write("Upload your medical documents and ask questions.")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload PDF", type="pdf")

docs = []
if uploaded_file is not None:
    loader = PyPDFLoader(uploaded_file.name)
    docs.extend(loader.load())

# -------------------------------
# Text Splitter
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# -------------------------------
# Retriever (BM25 only)
# -------------------------------
retriever = BM25Retriever.from_documents(splits, k=3)

# -------------------------------
# LLM + Prompt
# -------------------------------
llm = Ollama(model="mistral")

qa_prompt = ChatPromptTemplate.from_template("""
You are a helpful medical assistant. 
Use the context below to answer the question. If the answer is not in the context, say you don’t know.

Context:
{context}

Question:
{question}
""")

parser = StrOutputParser()

# -------------------------------
# Chain Fix (patched)
# -------------------------------
context = retriever  # BM25 retriever result

chain = (
    {
        "context": retriever,                  # will fetch docs
        "question": RunnablePassthrough(),     # forwards user input
    }
    | qa_prompt
    | llm
    | parser
)

# -------------------------------
# User Input
# -------------------------------
q = st.text_input("💬 Ask a medical question:")

if q:
    ans = chain.invoke(q)
    st.subheader("🧠 Answer")
    st.write(ans)
