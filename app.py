"""
TalentIQ — RAG-Powered HR & Talent Assessment Assistant
Streamlit UI: upload HR policy / job description / competency documents,
build a vector index, and ask grounded questions with source citations.
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.dirname(__file__))

from src.ingest import build_vectorstore, load_vectorstore, CHROMA_DIR
from src.rag_chain import answer_question
from src.llm import DeepSeekLLM

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="TalentIQ", page_icon="🧠", layout="wide")

st.title("🧠 TalentIQ")
st.caption("RAG-Powered HR & Talent Assessment Assistant — semantic search over policy docs, JDs, and competency frameworks.")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer, sources)

# ---------------------------------------------------------------------------
# Sidebar: document upload + index management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload HR policy docs, JDs, or competency frameworks",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for f in uploaded_files:
            save_path = os.path.join(DATA_DIR, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to the data folder.")

    existing_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".pdf", ".txt"))]
    st.write(f"**{len(existing_files)} document(s) ready to index:**")
    for f in existing_files:
        st.write(f"- {f}")

    if st.button("🔨 Build / Rebuild Index", type="primary", disabled=len(existing_files) == 0):
        with st.spinner("Chunking, embedding, and indexing documents..."):
            st.session_state.vectordb = build_vectorstore(DATA_DIR)
        st.success("Index built successfully.")

    if st.button("📂 Load Existing Index"):
        if os.path.exists(CHROMA_DIR):
            st.session_state.vectordb = load_vectorstore()
            st.success("Loaded existing index from disk.")
        else:
            st.warning("No existing index found. Build one first.")

    st.divider()
    st.header("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=4)

# ---------------------------------------------------------------------------
# Main: chat interface
# ---------------------------------------------------------------------------
if st.session_state.vectordb is None:
    st.info("👈 Upload documents and build the index from the sidebar to get started.")
else:
    question = st.chat_input("Ask a question about HR policy, job roles, or competency frameworks...")

    if question:
        try:
            llm = DeepSeekLLM()
        except ValueError as e:
            st.error(str(e))
            st.stop()

        with st.spinner("Retrieving relevant context and generating answer..."):
            answer, sources = answer_question(question, st.session_state.vectordb, llm, k=top_k)

        st.session_state.chat_history.append((question, answer, sources))

    for q, a, sources in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
            with st.expander(f"📚 View {len(sources)} source excerpt(s)"):
                for i, s in enumerate(sources, start=1):
                    st.markdown(f"**Source {i}: `{s.metadata.get('source', 'unknown')}`**")
                    st.text(s.page_content[:500] + ("..." if len(s.page_content) > 500 else ""))
                    st.divider()
