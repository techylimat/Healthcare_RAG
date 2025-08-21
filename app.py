import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

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
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from openai import OpenAI  # not needed now
# client = OpenAI()

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
mod
