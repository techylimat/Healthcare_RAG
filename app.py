import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.retrievers import BM25Retriever
from langchain.schema import BaseMessage  # for safe response handling

# ---------------------
# Sidebar settings (kept intact)
# ---------------------
st.sidebar.title("Settings")

top_k = st.sidebar.slider("Number of retrieved documents (k)", 1, 10, 5)

retriever_option = st.sidebar.radio(
    "Choose retriever",
    ["BM25 retriever"],  # Only BM25 remains
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
# Streamlit App (kept name the same)
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

            # Always BM25 now
            retriever = BM25Retriever.from_documents(docs)
            retriever.k = top_k

            st.session_state["retriever"] = retriever
            st.success("✅ Knowledge base created using BM25!")
        except Exception as e:
            st.error(f"Failed to load and index webpage: {e}")
            st.session_state["retriever"] = None
else:
    st.session_state["retriever"] = None

query = st.text_input("Ask a question about the webpage:")

if query and st.session_state.get("retriever"):
    with st.spinner("Generating answer..."):
        try:
            relevant_docs = st.session_state["retriever"].get_relevant_documents(query) or []
            if not relevant_docs:
                st.warning("No relevant documents found for this query.")
                st.stop()

            context = "\n".join([doc.page_content for doc in relevant_docs if getattr(doc, "page_content", None)])

            prompt = (
                "Answer the following question using the provided context.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            )
            raw_response = llm.invoke(prompt)

            # Normalize response to plain string
            if isinstance(raw_response, BaseMessage):
                response_text = raw_response.content
            elif isinstance(raw_response, dict) and "generated_text" in raw_response:
                response_text = raw_response["generated_text"]
            else:
                response_text = str(raw_response)

            st.subheader("Answer")
            st.write(response_text)

            with st.expander("Retrieved Context"):
                st.write(context)

        except Exception as e:
            st.error(f"Error during retrieval/answer generation: {e}")
elif query:
    st.warning("⚠️ Please load a webpage first.")
