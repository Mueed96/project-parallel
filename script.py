import os
import warnings
import logging
from datetime import datetime

import streamlit as st

from dotenv import load_dotenv; load_dotenv()
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Smart PDF Assistant",
    page_icon="📄",
    layout="centered"
)

# --- Custom CSS (unique colors & fonts) ---
st.markdown("""
    <style>
    body {
        background-color: #f0f8ff;
        color: #222222;
        font-family: 'Helvetica', sans-serif;
    }
    .main {
        background: #ffffff;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stApp h1 {
        text-align: center;
        color: #003366;
    }
    input {
        background: #ffffff !important;
        color: #000000 !important;
    }
    .stChatMessage[data-testid="stChatMessage-user"] {
        background: #cfe2f3 !important;
        border-radius: 6px;
        padding: 8px;
    }
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background: #d9ead3 !important;
        border-radius: 6px;
        padding: 8px;
    }
    .stDownloadButton button {
        background-color: #00509e;
        color: #ffffff !important;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.title("📄 Smart PDF Assistant")

# --- Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# --- Load PDF & Index ---
@st.cache_resource
def build_vectorstore():
    with st.spinner("Indexing your PDF..."):
        loader = PyPDFLoader("document.pdf")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        index = VectorstoreIndexCreator(
            embedding=embeddings,
            text_splitter=splitter
        ).from_loaders([loader])
        st.success("PDF indexed successfully!")
        return index.vectorstore

# --- User Input ---
query = st.chat_input("Type your question here...")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.chat_history.append({"role": "user", "content": query})

    groq_prompt = ChatPromptTemplate.from_template(
        "You are a knowledgeable assistant. Answer this question concisely and accurately: {user_input}"
    )

    groq_llm = ChatGroq(
        groq_api_key=st.secrets["groq"]["api_key"],
        model_name="llama3-8b-8192"
    )

    try:
        store = build_vectorstore()
        qa = RetrievalQA.from_chain_type(
            llm=groq_llm,
            chain_type="stuff",
            retriever=store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False
        )
        result = qa({"query": query})
        answer = result["result"]

        st.chat_message("assistant").markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"Oops! Something went wrong: {str(e)}")

# --- Download Chat History ---
if st.session_state.get("chat_history"):
    def export_history():
        log = []
        for entry in st.session_state.chat_history:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.append(f"[{timestamp}] {entry['role'].capitalize()}: {entry['content']}")
        return "\n\n".join(log)

    st.download_button(
        label="📥 Download Chat Log",
        data=export_history(),
        file_name="chat_log.txt",
        mime="text/plain"
    )
