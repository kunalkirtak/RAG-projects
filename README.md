# 🔎 RAG-Projects

**A portfolio of production-style Retrieval-Augmented Generation (RAG) systems** — from a single-PDF chatbot to a multi-document research agent with hybrid search, evaluation metrics, REST APIs, and Docker/CI deployment.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4?logo=google&logoColor=white)
![VectorDB](https://img.shields.io/badge/Vector%20DB-FAISS-4B8BBE)
![Framework](https://img.shields.io/badge/Framework-LangChain-1C3C3C)
![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![API](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Each sub-project builds on the last in complexity — same core RAG concepts (chunking, embeddings, vector search, grounded generation), increasing scope: single document → multi-format knowledge base → multi-document agent with hybrid retrieval, memory, and evaluation.

---

## 📁 Projects

| # | Project | What it does | Core additions |
|---|---|---|---|
| 01 | [**PDF Chatbot**](./01-PDF-Chatbot) | Upload a single PDF and ask natural-language questions, with cited source chunks. | FAISS + `sentence-transformers` embeddings, Gemini-grounded answers, CLI + Streamlit UI, Docker |
| 02 | [**Company Knowledge Assistant**](./02-company-knowledge-assistant) | Enterprise-style assistant over multiple internal documents (PDF/DOCX/TXT). | FastAPI REST backend, conversation memory, "answer only from context" guardrail, source citations |
| 03 | [**Multi-Document Research Agent**](./03-Multi-Document-Research-Agent) | Query across many documents at once with hybrid search and an evaluation harness. | FAISS + BM25 fused via Reciprocal Rank Fusion, query rewriting, Precision@K / Recall@K / MRR eval, unit tests, GitHub Actions CI |

Every project folder has its own detailed README covering architecture, setup, and usage — click through above for the full write-up.

---

## 🧠 What this repo demonstrates

- **End-to-end RAG pipeline design** — ingestion → chunking → embedding → vector indexing → retrieval → grounded generation
- **Retrieval strategies** — pure vector search (FAISS) as well as hybrid vector + keyword (BM25) search with rank fusion
- **Reliability engineering** — source citations, context-grounded prompts, and hallucination guardrails
- **Productionization** — REST APIs (FastAPI), chat UIs (Streamlit), Docker/Docker Compose, CI (GitHub Actions), structured logging
- **Evaluation** — retrieval quality metrics (Precision@K, Recall@K, MRR) and latency benchmarking
- **Code quality** — modular src layout, type hints, docstrings, and unit tests (pytest) in the most advanced project

## 🛠️ Common Tech Stack

| Layer | Tools used across projects |
|---|---|
| LLM | Google Gemini (`google-generativeai` / `google-genai`) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Search | FAISS, with BM25 hybrid fusion in project 03 |
| Orchestration | LangChain (text splitting, chains) |
| Document Parsing | PyMuPDF, `pypdf`, `python-docx` |
| Backend | FastAPI |
| Frontend | Streamlit |
| Infra | Docker, Docker Compose, GitHub Actions |
| Testing | pytest |

## 📂 Repository Structure

```
RAG-projects/
├── 01-PDF-Chatbot/                    # Single-PDF RAG chatbot
├── 02-company-knowledge-assistant/    # Multi-format enterprise knowledge assistant
├── 03-Multi-Document-Research-Agent/  # Hybrid-search multi-document research agent
├── LICENSE
└── README.md
```

## 🚀 Getting Started

Each project is self-contained with its own dependencies and `.env`/API key setup.

```bash
git clone https://github.com/kunalkirtak/RAG-projects.git
cd RAG-projects/<project-folder>

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Then follow the setup and usage instructions in that project's own `README.md` (API keys, running the CLI/API/Streamlit app, Docker, etc.).

## 📜 License

This repository is licensed under the [MIT License](./LICENSE).

## 👤 Author

**Kunal Kirtak**
GitHub: [@kunalkirtak](https://github.com/kunalkirtak)

If you find this useful, consider ⭐ starring the repo!
