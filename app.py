import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict
from langchain.schema import Document
import tempfile

# -------------------------------
# Streamlit UI (UNCHANGED)
# -------------------------------
st.set_page_config(page_title="Healthcare RAG", page_icon="🩺", layout="wide")
st.title("🩺 Healthcare RAG Chatbot")
st.write("Upload your medical documents and ask questions.")

# -------------------------------
# File Upload (safe temp-file)
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload PDF", type="pdf")

docs: List[Document] = []
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        loader = PyPDFLoader(tmp_file.name)
        docs.extend(loader.load())

# -------------------------------
# Chunking
# -------------------------------
splits: List[Document] = []
if docs:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

# -------------------------------
# Indexing & Retrieval
#   - Dense: FAISS + HuggingFaceEmbeddings
#   - Sparse: BM25
#   - Hybrid: simple RRF-style fusion
# -------------------------------
dense_retriever = None
sparse_retriever = None

if splits:
    # Dense index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(splits, embeddings)
    dense_retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # Sparse index (BM25)
    sparse_retriever = BM25Retriever.from_documents(splits, k=4)

def rrf_fuse(results_lists: List[List[Document]], k: int = 4, c: int = 60) -> List[Document]:
    """Reciprocal Rank Fusion for small lists of Documents."""
    score_map: Dict[str, float] = {}
    doc_by_id: Dict[str, Document] = {}

    for results in results_lists:
        for rank, d in enumerate(results):
            # Use id if present; else fallback to (page_content start + metadata hash)
            did = getattr(d, "id", None) or (d.page_content[:80] + str(hash(frozenset(d.metadata.items()))))
            doc_by_id[did] = d
            score_map[did] = score_map.get(did, 0.0) + 1.0 / (c + rank + 1)

    ranked_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)
    return [doc_by_id[i] for i in ranked_ids[:k]]

def hybrid_retrieve(query: str) -> List[Document]:
    if dense_retriever and sparse_retriever:
        dense = dense_retriever.get_relevant_documents(query)
        sparse = sparse_retriever.get_relevant_documents(query)
        return rrf_fuse([dense, sparse], k=4)
    elif dense_retriever:
        return dense_retriever.get_relevant_documents(query)
    elif sparse_retriever:
        return sparse_retriever.get_relevant_documents(query)
    else:
        return []

# -------------------------------
# LLM + Prompt
# -------------------------------
llm = Ollama(model="mistral")

qa_prompt = ChatPromptTemplate.from_template("""
You are a helpful medical assistant.
Use ONLY the context to answer. If the answer is not in the context, say you don’t know.

Context:
{context}

Question:
{question}
""".strip())

parser = StrOutputParser()

def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "No relevant context found."
    return "\n\n---\n\n".join(d.page_content for d in docs)

# -------------------------------
# Chain
# -------------------------------
retrieval_runnable = RunnableLambda(hybrid_retrieve)

chain = (
    {
        "context": retrieval_runnable | format_docs,
        "question": RunnablePassthrough(),
    }
    | qa_prompt
    | llm
    | parser
)

# -------------------------------
# User Input (UNCHANGED)
# -------------------------------
q = st.text_input("💬 Ask a medical question:")

if q:
    if not splits:
        st.warning("Please upload a PDF first so I can build the knowledge base.")
    else:
        try:
            ans = chain.invoke(q)
            st.subheader("🧠 Answer")
            st.write(ans)
        except Exception as e:
            st.error(f"Something went wrong while generating the answer: {e}")
