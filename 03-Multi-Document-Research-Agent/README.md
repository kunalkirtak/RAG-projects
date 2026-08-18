#  Multi-Document Research Agent

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue?logo=githubactions&logoColor=white)](.github/workflows/python.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_3.5_Flash-orange)](https://ai.google.dev/)
[![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-purple)](https://github.com/facebookresearch/faiss)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

An enterprise-grade **Retrieval-Augmented Generation (RAG)** system for querying across
multiple PDF, DOCX, and TXT documents at once — with hybrid (vector + keyword) search,
conversational memory, source citations, and a Gemini-powered answer engine.

Built to go beyond a "toy PDF chatbot": modular architecture, persistence, incremental
indexing, evaluation harness, REST API, chat UI, Docker packaging, and CI.

---

##  Features

| Category | Capabilities |
|---|---|
| **Ingestion** | Multi-file PDF / DOCX / TXT upload, automatic loading, text cleaning, metadata extraction |
| **Chunking** | Recursive & semantic (sentence-aware) strategies, configurable size/overlap |
| **Retrieval** | FAISS vector search + BM25 keyword search fused via Reciprocal Rank Fusion, metadata filtering, near-duplicate removal, context compression |
| **Generation** | Gemini `3.5-flash`, streaming responses, query rewriting for follow-ups, conversation memory, inline citations |
| **Interfaces** | CLI (`main.py`), REST API (`api.py`, FastAPI + Swagger), Chat UI (`streamlit_app.py`) |
| **Ops** | Structured logging, Docker + docker-compose, GitHub Actions CI, persistence & incremental indexing |
| **Evaluation** | Precision@K, Recall@K, MRR, latency benchmarking, embedding-space & timing visualizations |
| **Quality** | Type hints, docstrings throughout, unit tests (pytest), PEP8-formatted |

---

##  Architecture

```
                         ┌──────────────────────────┐
                         │   Documents (PDF/DOCX/TXT) │
                         └─────────────┬────────────┘
                                        ▼
                         ┌──────────────────────────┐
                         │   document_loader.py      │  clean + extract metadata
                         └─────────────┬────────────┘
                                        ▼
                         ┌──────────────────────────┐
                         │   chunker.py              │  recursive / semantic split
                         └─────────────┬────────────┘
                                        ▼
                         ┌──────────────────────────┐
                         │   embedder.py             │  sentence-transformers
                         └─────────────┬────────────┘
                                        ▼
                         ┌──────────────────────────┐
                         │   vector_store.py (FAISS) │  persisted, incremental
                         └─────────────┬────────────┘
                                        │
      ┌─────────────────────────────────┼─────────────────────────────────┐
      ▼                                                                   ▼
┌───────────────┐                                              ┌───────────────────┐
│ reranker.py    │  BM25 index + Reciprocal Rank Fusion         │ retriever.py       │
│ (lexical side) │◄─────────────────────────────────────────────┤ (hybrid orchestr.) │
└───────────────┘                                              └─────────┬─────────┘
                                                                          ▼
                                        ┌────────────────────────────────────────────┐
                                        │  dedup → metadata filter → context compress │
                                        └───────────────────────┬────────────────────┘
                                                                 ▼
                                        ┌────────────────────────────────────────────┐
                                        │  prompts.py + memory.py → gemini_client.py  │
                                        └───────────────────────┬────────────────────┘
                                                                 ▼
                                                   ┌─────────────────────────┐
                                                   │  Answer + Citations      │
                                                   └─────────────────────────┘

                     Exposed via:  main.py (CLI)  ·  api.py (FastAPI)  ·  streamlit_app.py (UI)
```

---

##  Screenshots

> _Add screenshots after running the app locally:_
![chat](assets/chat.png)


---

##  Google Colab Instructions

1. Open the notebook `03_Multi_Document_Research_Agent.ipynb` in Google Colab.
2. Run every cell top-to-bottom — this **generates the entire project** on disk
   (`os.makedirs` + file writes), installs dependencies, builds a sample vector
   database, and demonstrates the CLI, API, and evaluation flows.
3. Paste your Gemini API key into the **Configuration** cell:
   ```python
   GEMINI_API_KEY = "PASTE_YOUR_API_KEY_HERE"
   ```
   Get a free key at https://aistudio.google.com/app/apikey.
4. The final cell zips the generated project so you can download it and push
   it straight to GitHub.

---

##  Local Setup

```bash
git clone https://github.com/kunalkirtak/03_Multi_Document_Research_Agent.git
cd 03_Multi_Document_Research_Agent

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env            # then edit .env and paste your GEMINI_API_KEY
```

### Docker

```bash
cp .env.example .env            # edit with your API key
docker compose up --build       # API on :8000, Streamlit UI on :8501
```

---

## ‍ Usage

### CLI

```bash
# Ingest every document in data/documents/
python main.py ingest --path data/documents

# Ask a single question
python main.py query "What is the refund policy?"

# Interactive chat session (with memory)
python main.py chat

# Run retrieval evaluation
python main.py evaluate
```

### REST API

```bash
uvicorn api:app --reload --port 8000
# Swagger UI:  http://localhost:8000/docs
```

| Method | Endpoint | Description |
|---|---|---|
| `GET`    | `/health`             | Liveness check |
| `GET`    | `/config`              | Current (redacted) configuration |
| `POST`   | `/upload`               | Upload & index one or more documents |
| `POST`   | `/query`                | Ask a question, get an answer + citations |
| `DELETE` | `/documents/{doc_id}`   | Remove a document from the index |
| `POST`   | `/reindex`               | Rebuild the index from `data/documents/` |

Example query:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval augmented generation?", "top_k": 5}'
```

### Streamlit Chat UI

```bash
streamlit run streamlit_app.py
```
Upload documents from the sidebar, adjust the vector/BM25 balance and Top-K live,
and chat with full source citations and response timing per turn.

---

##  Folder Structure

```
03_Multi_Document_Research_Agent/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── setup.py
├── main.py                 # CLI entry point
├── api.py                  # FastAPI REST service
├── streamlit_app.py        # Streamlit chat UI
├── configs/
│   └── config.yaml
├── data/
│   ├── documents/           # Source documents (gitignored contents)
│   └── vectorstore/         # Persisted FAISS index (gitignored contents)
├── logs/                    # Timestamped log files
├── tests/                   # pytest unit tests
├── assets/                  # Screenshots / generated charts
├── notebooks/                # This Colab notebook
├── src/
│   ├── config.py             # Configuration system
│   ├── logger.py              # Centralised logging
│   ├── utils.py                # Shared helpers
│   ├── document_loader.py       # PDF / DOCX / TXT loading + cleaning
│   ├── chunker.py                # Recursive & semantic chunking
│   ├── embedder.py                # SentenceTransformers wrapper
│   ├── vector_store.py             # FAISS persistence & incremental indexing
│   ├── reranker.py                  # BM25 + Reciprocal Rank Fusion
│   ├── retriever.py                  # Hybrid retrieval orchestration
│   ├── memory.py                      # Conversation memory
│   ├── prompts.py                      # Prompt templates
│   ├── gemini_client.py                 # Gemini API wrapper (retry + streaming)
│   ├── pipeline.py                       # End-to-end RAG orchestrator
│   ├── evaluation.py                      # Precision@K / Recall@K / latency
│   └── visualization.py                    # Plotly charts
└── .github/workflows/python.yml              # CI: lint + test
```

---

##  Technologies

Python 3.11 · Google Gemini API (`gemini-3.5-flash`) · FAISS · sentence-transformers ·
rank-bm25 · FastAPI · Streamlit · Pydantic · pandas / numpy / scikit-learn · Plotly ·
pypdf · python-docx · Docker

---

##  Performance

Measured on the bundled example dataset (small corpus, CPU-only, `all-MiniLM-L6-v2`):

| Metric | Typical value |
|---|---|
| Embedding (per chunk) | ~5–15 ms |
| Vector search (FAISS, flat index) | < 5 ms |
| BM25 search | < 5 ms |
| End-to-end query (retrieval + generation) | 1–3 s (dominated by the Gemini API call) |

Run `python main.py evaluate` or the notebook's **Evaluation** section to benchmark
against your own labelled query set and generate latency/embedding visualizations.

---

##  Future Improvements

- Swap `IndexFlatIP` for an approximate index (`IndexIVFFlat` / `IndexHNSWFlat`) for
million-scale corpora.
- Add a real cross-encoder reranker (e.g. `ms-marco-MiniLM`) as a drop-in alternative
to the current RRF-based fusion.
- Multi-tenant document namespaces / access control in the API.
- Async ingestion queue for very large document batches.
- Structured output (JSON mode) for downstream tool integrations.

---

##  License

Released under the [MIT License](LICENSE).

##  Author

Built as a portfolio-ready reference implementation of a production RAG system.
Contributions and issues welcome.
