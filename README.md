# TalentIQ — RAG-Powered HR & Talent Assessment Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions over HR policy
documents, job descriptions, and competency frameworks — with source citations for
every answer.

## Why this project

Built to demonstrate hands-on experience with the modern GenAI stack that's
increasingly asked for in Software Engineer / Data Scientist / AI Engineer roles:
LangChain orchestration, vector databases (ChromaDB), LLM API integration, and a
usable UI (Streamlit) — all wired together into one working, demoable system.

## Tech Stack

- **LangChain** — document loading, chunking, and retrieval orchestration
- **ChromaDB** — persistent vector database for semantic search
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embeddings (no API cost)
- **DeepSeek API** — LLM for grounded answer generation
- **Streamlit** — interactive web UI
- **Python** — core language throughout

## How it works

1. Upload HR policy PDFs/TXT files through the sidebar
2. Click **Build Index** — documents are chunked (800 chars, 120 overlap) and
   embedded into a persistent ChromaDB collection
3. Ask a question in the chat box — the system retrieves the top-k most relevant
   chunks, builds a grounded prompt, and calls DeepSeek to generate an answer
4. Every answer includes an expandable **source citation** panel showing exactly
   which document chunks were used

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/Charitha-Pendyala/TalentIQ.git
cd TalentIQ
pip install -r requirements.txt
```

### 2. Configure your DeepSeek API key

```bash
cp .env.example .env
# Edit .env and add your actual DEEPSEEK_API_KEY
```

Get a key at [platform.deepseek.com](https://platform.deepseek.com).

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Two sample documents (a leave policy and
a competency framework) are already included in `data/` so you can try it
immediately — just click **Build Index** in the sidebar, then ask something like:

- *"How many days of annual leave am I entitled to, and does it carry forward?"*
- *"What's expected of a Level 1 Associate Software Engineer?"*
- *"Can I work fully remote?"*

## Project Structure

```
TalentIQ/
├── app.py                  # Streamlit UI
├── src/
│   ├── ingest.py           # Document loading, chunking, embedding, ChromaDB storage
│   ├── rag_chain.py        # Retrieval + prompt construction + generation
│   └── llm.py              # DeepSeek API client wrapper
├── data/                   # Sample HR documents (add your own PDFs/TXT here)
├── requirements.txt
└── .env.example
```

## Notes on design choices

- **Local embeddings, API-only generation**: embeddings run locally via
  sentence-transformers (free, fast, no API cost per document), while only the
  final answer generation calls the DeepSeek API — this keeps ingestion cheap
  even for large document sets.
- **Source citations aren't cosmetic**: every retrieved chunk is tagged with its
  source filename at ingestion time, so answers can always be traced back to a
  specific document — important for HR use cases where trust and auditability
  matter.
- **Grounded-only answering**: the system prompt explicitly instructs the model to
  say when an answer isn't in the provided context, rather than hallucinating from
  general knowledge — a core RAG-quality concern.

## Possible extensions

- Swap ChromaDB for FAISS (already in requirements) for larger-scale deployments
- Add conversation memory for multi-turn follow-up questions
- Add a simple evaluation harness (e.g., a set of Q&A pairs with expected sources)
  to measure retrieval quality over time
