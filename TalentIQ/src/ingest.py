"""
Ingestion pipeline for TalentIQ.

Loads HR policy / job description / competency framework documents (PDF or TXT),
splits them into overlapping chunks, embeds them with a local sentence-transformers
model (so no extra API cost for embeddings), and stores them in a persistent
ChromaDB collection for semantic retrieval.
"""

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_store")
COLLECTION_NAME = "talentiq_docs"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(data_dir: str) -> List[Document]:
    """Load every .pdf and .txt file in data_dir into LangChain Document objects."""
    docs: List[Document] = []
    for fname in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, fname)
        if fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            loaded = loader.load()
        elif fname.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            loaded = loader.load()
        else:
            continue

        # Tag each chunk with its originating filename so we can cite sources later.
        for d in loaded:
            d.metadata["source"] = fname
        docs.extend(loaded)

    return docs


def split_documents(docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 120) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


def build_vectorstore(data_dir: str, persist_dir: str = CHROMA_DIR) -> Chroma:
    """Full ingestion: load -> split -> embed -> persist to Chroma."""
    raw_docs = load_documents(data_dir)
    if not raw_docs:
        raise ValueError(f"No .pdf or .txt files found in {data_dir}")

    chunks = split_documents(raw_docs)
    embeddings = get_embeddings()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    vectordb.persist()
    return vectordb


def load_vectorstore(persist_dir: str = CHROMA_DIR) -> Chroma:
    """Load an already-built Chroma store from disk (no re-embedding needed)."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    print(f"Ingesting documents from: {data_dir}")
    vs = build_vectorstore(data_dir)
    print(f"Vector store built and persisted to: {CHROMA_DIR}")
