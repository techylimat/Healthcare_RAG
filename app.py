import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# ---------------------
# Sidebar settings
# ---------------------
st.sidebar.title("Settings")

top_k = st.sidebar.slider("Number of retrieved documents (k)", 1, 10, 5)

retriever_option = st.sidebar.radio(
    "Choose retriever",
    ["Chroma (default)", "BM25 retriever", "LLM reranker"],
    index=0
)

# ---------------------
# HuggingFace model setup
# ---------------------
hf_model_id = "mistralai/Mistral-7B-Instruct-v0.3"
hf_api_key = st.secrets.get("HF_TOKEN")

if not hf_api_key:
    st.error("Please set your HuggingFace API token in Streamlit secrets.")
    st.stop()

llm = HuggingFaceEndpoint(
    repo_id=hf_model_id,
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=hf_api_key,
)

# ---------------------
# Embeddings
# ---------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------------
# Context function (retrievers)
# ---------------------
def make_context_fn(vs, k, retriever_option):
    if not vs:
        return None

    retriever = vs.as_retriever(search_kwargs={"k": k})

    if retriever_option == "BM25 retriever":
        bm25 = BM25Retriever.from_documents(vs.get())
        return bm25

    elif retriever_option == "LLM reranker":
        try:
            compressor = LLMChainExtractor.from_llm(llm)
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever
            )
        except Exception as e:
            st.warning(f"⚠️ LLM reranker unavailable ({type(e).__name__}: {e}). Falling back to Chroma retriever.")
            return retriever

    return retriever  # default: Chroma


# ---------------------
# Streamlit App
# ---------------------
st.title("📖 RAG with Chroma / BM25 / LLM Reranker")

url = st.text_input("Enter a webpage URL to load knowledge:")

if url:
    with st.spinner("Loading and indexing webpage..."):
        try:
            loader = WebBaseLoader(url)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = text_splitter.split_documents(data)

            vs = Chroma.from_documents(docs, embeddings)
            st.success("✅ Knowledge base created!")
        except Exception as e:
            st.error(f"Failed to load and index webpage: {e}")
            vs = None
else:
    vs = None

ctx_fn = make_context_fn(vs, k=top_k, retriever_option=retriever_option) if vs else None

query = st.text_input("Ask a question about the webpage:")

if query and ctx_fn:
    with st.spinner("Generating answer..."):
        try:
            relevant_docs = ctx_fn.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in relevant_docs])

            prompt = f"Answer the following question using the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
            response = llm.invoke(prompt)

            st.subheader("Answer")
            st.write(response)

            with st.expander("Retrieved Context"):
                st.write(context)

        except Exception as e:
            st.error(f"Error during retrieval/answer generation: {e}")
elif query:
    st.warning("⚠️ Please load a webpage first.")
