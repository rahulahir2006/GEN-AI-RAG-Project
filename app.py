"""
Streamlit RAG Chat App
----------------------
A deployable Streamlit front-end for the Chroma + Gemini RAG pipeline
defined in createdb.py / main.py.

Run with:
    streamlit run app.py

Requires a .env file (or Streamlit secrets) with GOOGLE_API_KEY set.
Get a free key at https://aistudio.google.com/apikey
"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass



import gc
import os
import chromadb
import shutil
import tempfile
import time
import warnings
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Harmless upstream google-genai SDK warning (fires even with no tools passed).
# See: https://github.com/googleapis/python-genai/issues/2902
warnings.filterwarnings("ignore", message=".*Automatic function calling.*")

load_dotenv()

CHROMA_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Document Q&A — RAG Chat",
    page_icon="📚",
    layout="wide",
)

# --------------------------------------------------------------------------
# Cached resources — loaded once per session/process
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, temperature: float):
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


@st.cache_resource(show_spinner=False)
def get_vectorstore(db_path: str, _embedding_model):
    """
    Keep one Chroma handle alive instead of creating a new one on every
    Streamlit rerun/query.  This is important on Windows because Chroma
    can keep data_level0.bin open.
    """
    if rebuild:
        clear_vectorstore_cache()
        if os.path.isdir(CHROMA_DIR):
            remove_chroma_dir(CHROMA_DIR)
os.makedirs(CHROMA_DIR, exist_ok=True)
persistent_client = chromadb.PersistentClient(path=CHROMA_DIR)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    client=persistent_client,
)


def clear_vectorstore_cache():
    """
    Release Streamlit's cached Chroma object before rebuilding/deleting the DB.
    gc.collect() helps Windows release file handles held by Python objects.
    """
    get_vectorstore.clear()
    gc.collect()
    time.sleep(0.5)


def remove_chroma_dir(db_path, retries=5):
    """
    Windows-safe Chroma directory removal.

    WinError 32 means another process/object still has a file open. We retry
    briefly after releasing cached resources. If it is still locked, raise a
    clear error instead of silently corrupting the database.
    """
    if not os.path.isdir(db_path):
        return

    last_error = None

    for attempt in range(retries):
        try:
            shutil.rmtree(db_path)
            return
        except PermissionError as e:
            last_error = e
            gc.collect()
            time.sleep(0.75 * (attempt + 1))

    raise PermissionError(
        f"Could not remove '{db_path}'. Chroma is still using one of its files. "
        "Stop any other Streamlit/Python process using this project and try "
        "Rebuild KB again."
    ) from last_error


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


class RateLimitedError(Exception):
    """Raised when the Gemini API is still rate-limited after all retries."""


def invoke_llm_with_backoff(llm, prompt, max_retries=3, base_delay=5):
    """Call llm.invoke, retrying on rate-limit errors with exponential backoff.

    Gemini's free tier has tight requests-per-minute caps. Rather than letting
    the raw ResourceExhausted error crash the whole page, retry a few times
    and then surface a clean, actionable message in the chat.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(prompt)
        except ResourceExhausted as e:
            last_error = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            raise RateLimitedError(
                "Gemini API rate limit reached even after retrying. "
                "Wait a minute before trying again, or check your usage "
                "at aistudio.google.com/apikey."
            ) from e
    raise RateLimitedError(str(last_error))


# --------------------------------------------------------------------------
# Ingestion helper — mirrors createdb.py but works on uploaded files
# --------------------------------------------------------------------------
def ingest_pdfs(uploaded_files, embedding_model, chunk_size=1000, chunk_overlap=200,
                 rebuild=False):
    all_docs = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for uf in uploaded_files:
            tmp_path = os.path.join(tmp_dir, uf.name)
            with open(tmp_path, "wb") as f:
                f.write(uf.getbuffer())
            loader = PyPDFLoader(tmp_path)
            all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(all_docs)

    if rebuild:
        # IMPORTANT: release the Chroma instance created during a previous
        # Streamlit run before Windows tries to delete data_level0.bin.
        clear_vectorstore_cache()

        if os.path.isdir(CHROMA_DIR):
            remove_chroma_dir(CHROMA_DIR)

    persistent_client = chromadb.PersistentClient(path=CHROMA_DIR)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client=persistent_client,
        )

    # The database has been persisted. Do not keep the ingestion handle alive:
    # on Windows that handle can keep data_level0.bin locked.
    del vectorstore
    gc.collect()
    time.sleep(0.25)

    # The next query will create/cache a fresh handle.
    clear_vectorstore_cache()

    return len(chunks)


# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources": [...]}

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = os.path.isdir(CHROMA_DIR) and bool(
        os.listdir(CHROMA_DIR)
    )

# --------------------------------------------------------------------------
# Sidebar — document management & settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    col_a, col_b = st.columns(2)
    add_clicked = col_a.button("➕ Add to KB", use_container_width=True)
    rebuild_clicked = col_b.button("🔄 Rebuild KB", use_container_width=True)

    if (add_clicked or rebuild_clicked) and not uploaded_files:
        st.warning("Please upload at least one PDF first.")

    if uploaded_files and (add_clicked or rebuild_clicked):
        embedding_model = get_embedding_model()
        with st.spinner("Processing PDF(s) and building embeddings..."):
            n_chunks = ingest_pdfs(
                uploaded_files,
                embedding_model,
                rebuild=rebuild_clicked,
            )
        st.session_state.vectorstore_ready = True
        st.success(f"Indexed {n_chunks} chunks from {len(uploaded_files)} file(s).")

    st.divider()
    st.header("⚙️ Settings")

    model_name = st.selectbox(
        "Gemini model",
        ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.subheader("Retriever (MMR)")
    k = st.slider("k (results returned)", 1, 10, 4)
    fetch_k = st.slider("fetch_k (candidates)", k, 30, max(10, k))
    lambda_mult = st.slider("λ (relevance ↔ diversity)", 0.0, 1.0, 0.5, 0.05)

    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        transcript = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages
        )
        st.download_button(
            "⬇️ Download chat",
            data=transcript,
            file_name=f"chat_{datetime.now():%Y%m%d_%H%M%S}.txt",
            use_container_width=True,
        )

# --------------------------------------------------------------------------
# Main chat interface
# --------------------------------------------------------------------------
st.title("📚 Chat with your Documents")
st.caption("RAG-powered Q&A over your PDFs — Chroma + BAAI embeddings + Gemini")

if not st.session_state.vectorstore_ready:
    st.info(
        "No knowledge base found yet. Upload a PDF and click **Add to KB** "
        "in the sidebar to get started."
    )
    st.stop()

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources used"):
                for i, src in enumerate(msg["sources"], 1):
                    page = src.metadata.get("page", "?")
                    source_file = src.metadata.get("source", "unknown")
                    st.markdown(f"**{i}. {os.path.basename(str(source_file))} — page {page}**")
                    st.caption(src.page_content[:400] + "...")

# Chat input
query = st.chat_input("Ask something about your document...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    embedding_model = get_embedding_model()
    vectorstore = get_vectorstore(CHROMA_DIR, embedding_model)
    llm = get_llm(model_name, temperature)

    if vectorstore is None:
        st.error("The knowledge base could not be opened. Please rebuild it.")
        st.stop()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )

    with st.chat_message("assistant"):
        docs = []
        answer_text = None
        try:
            with st.spinner("Searching document and generating answer..."):
                docs = retriever.invoke(query)
                context = "\n\n".join(doc.page_content for doc in docs)
                final_prompt = PROMPT.invoke({"context": context, "question": query})
                response = invoke_llm_with_backoff(llm, final_prompt)
                answer_text = response.content
        except RateLimitedError as e:
            st.error(f"⏳ {e}")
            answer_text = (
                "⚠️ I couldn't generate an answer — the Gemini API rate limit "
                "was reached. Please wait a moment and try again."
            )
        except GoogleAPICallError as e:
            st.error(f"API error: {e}")
            answer_text = "⚠️ I couldn't generate an answer due to an API error. Please try again."

        st.markdown(answer_text)
        if docs:
            with st.expander("📄 Sources used"):
                for i, src in enumerate(docs, 1):
                    page = src.metadata.get("page", "?")
                    source_file = src.metadata.get("source", "unknown")
                    st.markdown(f"**{i}. {os.path.basename(str(source_file))} — page {page}**")
                    st.caption(src.page_content[:400] + "...")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "sources": docs}
    )