# 📄 PDF Chatbot using Retrieval-Augmented Generation (RAG)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Gemini](https://img.shields.io/badge/LLM-Gemini%203.5%20Flash-orange)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An enterprise-style RAG chatbot that lets you upload a PDF and ask natural
language questions about it. Answers are grounded in the document using
retrieval-augmented generation, with cited source chunks for every answer.

---

## Overview

The pipeline extracts text from a PDF, splits it into overlapping chunks,
embeds each chunk locally with `all-MiniLM-L6-v2`, indexes the vectors in a
FAISS store, and at query time retrieves the top-k most similar chunks to
ground a Gemini 3.5 Flash answer. Both a CLI demo and a full Streamlit app
are included.

## Architecture

```
PDF --> PyMuPDF extraction --> RecursiveCharacterTextSplitter (chunking)
     --> SentenceTransformer embeddings --> FAISS index (persisted to disk)

User question --> embed query --> FAISS similarity search --> top-k chunks
               --> prompt template --> Gemini 3.5 Flash --> answer + sources
```

## Folder Structure

```
01-PDF-Chatbot/
├── app.py                  # Streamlit web application
├── build_vector_store.py   # Builds and persists the FAISS index
├── chunk_documents.py       # Recursive chunking of ingested PDFs
├── ingest.py                # PDF text extraction
├── search_demo.py           # CLI similarity-search demo
├── requirements.txt
├── Dockerfile
├── LICENSE
├── .gitignore
├── config/
│   └── config.py            # Chunking, retrieval & model settings
├── core/
│   ├── logger.py
│   ├── utils.py
│   ├── loader.py
│   ├── chunker.py
│   ├── embedding.py         # all-MiniLM-L6-v2 wrapper
│   ├── vector_store.py      # FAISS build / save / load / search
│   ├── prompt.py            # RAG prompt template
│   ├── gemini_client.py     # Gemini 3.5 Flash wrapper
│   └── rag_pipeline.py      # End-to-end RAG orchestration
├── data/
│   ├── raw/                 # Uploaded PDFs
│   ├── processed/           # Extracted text (JSON)
│   └── chunks/               # Chunked text (JSON)
├── vector_store/             # faiss_index.bin + metadata.json
├── logs/
└── screenshots/
```

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Google Gemini 3.5 Flash (`google-generativeai`) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector DB | FAISS (`faiss-cpu`) |
| PDF parsing | PyMuPDF |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| UI | Streamlit |
| Infra | Docker |

## Installation

```bash
git clone https://github.com/<your-username>/01-PDF-Chatbot.git
cd 01-PDF-Chatbot

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

export GEMINI_API_KEY="your-api-key-here"   # Windows: set GEMINI_API_KEY=...
```

## Usage

**1. Ingest a PDF and build the index**

```bash
cp /path/to/your.pdf data/raw/
python ingest.py
python chunk_documents.py
python build_vector_store.py
```

**2. Ask questions from the CLI**

```bash
python search_demo.py
```

**3. Or launch the full web app**

```bash
streamlit run app.py
```

Then open the sidebar to upload a PDF and start chatting. Every answer
expands to show the retrieved source chunks with similarity scores.

## Run with Docker

```bash
docker build -t pdf-chatbot .
docker run -p 8501:8501 -e GEMINI_API_KEY=your-api-key-here pdf-chatbot
```

## Example Output

```
Question: What is the termination clause in this contract?

Answer: Either party may terminate the agreement with 30 days' written
notice, or immediately in the event of a material breach that remains
uncured for 15 days after notice.

Sources:
[1] contract.pdf — page 4 — score 0.812
[2] contract.pdf — page 5 — score 0.774
```

## Screenshots

> _Add screenshots of the Streamlit UI here, e.g._
> `![Chat UI](screenshots/chat_ui.png)`

## Future Improvements

- Swap FAISS `IndexFlatIP` for an approximate index (HNSW/IVF) for large corpora
- Support multi-PDF collections with per-document filtering
- Stream Gemini responses token-by-token in the UI
- Add automated evaluation (retrieval precision/recall, answer faithfulness)
- Re-rank retrieved chunks with a cross-encoder before prompting
- Persist chat history per user session (SQLite/Postgres)

## License

Released under the [MIT License](LICENSE).\n
