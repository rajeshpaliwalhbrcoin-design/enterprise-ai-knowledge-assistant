import streamlit as st
import ollama
import pandas as pd
import os
import base64
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

# ---------------- CONFIG ----------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")

DB_NAME = os.path.join(DATA_DIR, "assistant_memory.db")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "vectors.index")

OLLAMA_MODEL = "llama3.2:latest"

SIMILARITY_THRESHOLD = 0.95
TOP_K = 4


# ---------------- PAGE CONFIG ----------------

def get_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

logo_path = os.path.join(ASSETS_DIR, "logo.png")
logo_b64 = get_base64(logo_path)

st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    layout="wide"
)


# ---------------- SPLASH SCREEN ----------------

if "splash_shown" not in st.session_state:
    splash = st.empty()

    splash.markdown(
        """
        <div style="position:fixed;inset:0;background:white;display:flex;flex-direction:column;align-items:center;justify-content:center;">
        <h2>Enterprise AI Knowledge Assistant</h2>
        <p>Loading AI Engine...</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    time.sleep(1)
    splash.empty()
    st.session_state.splash_shown = True


# ---------------- EMBEDDINGS ----------------

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-mpnet-base-v2")

embedder = load_embedder()


# ---------------- DATABASE ----------------

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS query_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()


# ---------------- CACHE CHECK ----------------

def check_cache(user_query):

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("SELECT question, answer FROM query_cache")
    rows = c.fetchall()

    conn.close()

    if not rows:
        return None

    questions = [r[0] for r in rows]
    answers = [r[1] for r in rows]

    query_emb = embedder.encode(user_query, convert_to_tensor=True)
    cache_embs = embedder.encode(questions, convert_to_tensor=True)

    hits = util.semantic_search(query_emb, cache_embs, top_k=1)

    if hits and hits[0]:
        best = hits[0][0]

        if best['score'] >= SIMILARITY_THRESHOLD:
            return answers[best['corpus_id']]

    return None


def save_cache(question, answer):

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute(
        "INSERT INTO query_cache (question,answer) VALUES (?,?)",
        (question, answer)
    )

    conn.commit()
    conn.close()


# ---------------- RETRIEVAL ----------------

def search_engine(query):

    if not os.path.exists(FAISS_INDEX_FILE):
        return ""

    index = faiss.read_index(FAISS_INDEX_FILE)

    query_vec = embedder.encode([query])
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    D, I = index.search(query_vec.astype("float32"), TOP_K)

    contexts = []

    for idx in I[0]:
        if idx == -1:
            continue

    return "\n".join(contexts)


# ---------------- LLM ----------------

def llm_engine(context, question):

    if context:

        prompt = f"""
You are an enterprise AI assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}
"""

    else:

        prompt = f"""
You are an enterprise AI assistant.

Answer the question clearly.

Question:
{question}
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    return response["message"]["content"]


# ---------------- UI ----------------

st.title("🤖 Enterprise AI Knowledge Assistant")


if "messages" not in st.session_state:

    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ---------------- CHAT INPUT ----------------

if prompt := st.chat_input("Ask a question..."):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    st.chat_message("user").write(prompt)

    cached = check_cache(prompt)

    if cached:

        answer = cached

    else:

        context = search_engine(prompt)

        answer = llm_engine(context, prompt)

        save_cache(prompt, answer)

    st.chat_message("assistant").write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )