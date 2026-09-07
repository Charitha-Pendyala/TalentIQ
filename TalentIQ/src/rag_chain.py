"""
Core RAG logic: given a user question, retrieve the most relevant chunks
from ChromaDB, build a grounded prompt, and call DeepSeek to generate an
answer. Returns both the answer and the source chunks used, so the UI
can show citations.
"""

from typing import List, Dict, Tuple
from langchain_core.documents import Document
from src.llm import DeepSeekLLM

SYSTEM_PROMPT = """You are TalentIQ, an HR policy and talent assessment assistant.
Answer the user's question using ONLY the provided context excerpts from HR policy
documents, job descriptions, and competency frameworks.

Rules:
- If the answer is not contained in the context, say so clearly instead of guessing.
- Be concise and structure multi-part answers with bullet points where helpful.
- When you use a specific fact, mention which source document it came from.
"""


def format_context(chunks: List[Document]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        parts.append(f"[Source {i}: {source}]\n{chunk.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(
    question: str,
    vectordb,
    llm: DeepSeekLLM,
    k: int = 4,
) -> Tuple[str, List[Document]]:
    """
    Retrieve top-k relevant chunks for the question, then ask the LLM to
    answer grounded in that context. Returns (answer_text, retrieved_chunks).
    """
    retrieved_chunks = vectordb.similarity_search(question, k=k)
    context = format_context(retrieved_chunks)

    user_prompt = f"""Context excerpts:

{context}

Question: {question}

Answer using only the context above. Cite the source document(s) you used."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    answer = llm.chat(messages)
    return answer, retrieved_chunks
