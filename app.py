import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from chromadb import PersistentClient  # ✅ New import

# --- Config
st.set_page_config(page_title="Healthcare RAG Bot", layout="wide")

# --- Sidebar
with st.sidebar:
    st.title("⚕️ Healthcare RAG Chatbot")
    st.markdown("Upload medical documents (PDF, DOCX, TXT) and ask questions.")
    api_key = st.text_input("Enter OpenAI API Key:", type="password")
    st.session_state["OPENAI_API_KEY"] = api_key

# --- Initialize LLM
llm = None
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- File uploader
uploaded_files = st.file_uploader(
    "Upload documents", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

doc_paths = []
if uploaded_files:
    for file in uploaded_files:
        path = os.path.join("uploads", file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        doc_paths.append(path)

# --- Process documents
documents = []
for path in doc_paths:
    if path.endswith(".pdf"):
        documents.extend(PyPDFLoader(path).load())
    elif path.endswith(".docx"):
        documents.extend(Docx2txtLoader(path).load())
    elif path.endswith(".txt"):
        documents.extend(TextLoader(path).load())

# --- Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# --- Vectorstore
embeddings = OpenAIEmbeddings()

vs = None
if st.button("Ingest"):
    if doc_paths:
        with st.spinner("Building vectorstore..."):
            client = PersistentClient(path=".chromadb")  # ✅ persistent storage
            vs = Chroma.from_documents(
                docs,
                embeddings,
                persist_directory="./chroma_store",
                client=client
            )
        st.success("Index updated successfully!")
    else:
        st.warning("Upload at least one document before ingestion.")

# --- QA Section
if vs:
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    retriever = ContextualCompressionRetriever(
        base_compressor=LLMChainExtractor.from_llm(llm), 
        base_retriever=vs.as_retriever()
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory
    )

    query = st.text_input("Ask a question about your documents:")
    if query:
        with st.spinner("Thinking..."):
            response = qa.invoke({"question": query})
            st.write(response["answer"])
