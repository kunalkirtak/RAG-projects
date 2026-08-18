"""
api.py
------
FastAPI REST API for the Multi-Document Research Agent.

Endpoints:
    GET    /health          - liveness check
    GET    /config           - current (redacted) configuration
    POST   /upload           - upload one or more documents and index them
    POST   /query             - ask a question
    DELETE /documents/{doc_id} - remove a document's chunks (requires reindex)
    POST   /reindex           - rebuild the vector store from data/documents

Run with:  uvicorn api:app --reload --port 8000
Swagger UI is auto-generated at /docs.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import config
from src.logger import get_logger
from src.pipeline import RAGPipeline

logger = get_logger(__name__)

app = FastAPI(
    title="Multi-Document Research Agent API",
    description="Hybrid-search RAG service over PDF / DOCX / TXT documents, powered by Gemini.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so unexpected errors return clean JSON instead of a raw 500 page."""
    logger.error("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})

_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Lazily instantiate the (expensive-to-load) pipeline as a singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(lazy_gemini=True)
    return _pipeline


class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's natural-language question")
    top_k: int = Field(default=5, ge=1, le=20)
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None)
    use_memory: bool = Field(default=True)


class SourceModel(BaseModel):
    citation: str
    filename: str
    chunk_index: Optional[int]
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceModel]
    rewritten_query: str
    timing: Dict[str, float]


class UploadResponse(BaseModel):
    num_documents: int
    num_chunks: int
    elapsed_seconds: float
    total_vectors: int


class HealthResponse(BaseModel):
    status: str
    total_vectors: int


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness/readiness probe."""
    pipeline = get_pipeline()
    return HealthResponse(status="ok", total_vectors=len(pipeline.vector_store))


@app.get("/config", tags=["system"])
def get_config() -> Dict[str, Any]:
    """Return the current (secret-redacted) configuration."""
    return config.to_dict()


@app.post("/upload", response_model=UploadResponse, tags=["documents"])
async def upload_documents(files: List[UploadFile] = File(...)) -> UploadResponse:
    """Upload one or more documents, save them to `data/documents/`, and index them."""
    pipeline = get_pipeline()
    saved_paths = []
    documents_dir = Path(config.documents_dir)
    documents_dir.mkdir(parents=True, exist_ok=True)

    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in {".pdf", ".docx", ".txt", ".md"}:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
        dest = documents_dir / upload.filename
        with open(dest, "wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        saved_paths.append(str(dest))

    stats = pipeline.ingest_paths(saved_paths)
    return UploadResponse(**stats)


@app.post("/query", response_model=QueryResponse, tags=["query"])
def query(request: QueryRequest) -> QueryResponse:
    """Ask a question against the indexed document collection."""
    pipeline = get_pipeline()
    if len(pipeline.vector_store) == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Call /upload or /reindex first.")

    result = pipeline.query(
        request.question,
        top_k=request.top_k,
        metadata_filter=request.metadata_filter,
        use_memory=request.use_memory,
    )
    return QueryResponse(
        answer=result["answer"],
        sources=[SourceModel(**s) for s in result["sources"]],
        rewritten_query=result["rewritten_query"],
        timing=result["timing"],
    )


@app.delete("/documents/{doc_id}", tags=["documents"])
def delete_document(doc_id: str) -> Dict[str, str]:
    """Remove all chunks belonging to `doc_id` and persist the updated index.

    Note: FAISS's IndexFlatIP does not support in-place deletion, so this
    rebuilds the index from the remaining records.
    """
    pipeline = get_pipeline()
    remaining = [r for r in pipeline.vector_store.records if r["doc_id"] != doc_id]
    if len(remaining) == len(pipeline.vector_store.records):
        raise HTTPException(status_code=404, detail=f"No chunks found for doc_id={doc_id}")

    from src.vector_store import VectorStore
    new_store = VectorStore(dimension=pipeline.embedder.dimension)
    if remaining:
        embeddings = pipeline.embedder.embed_texts([r["text"] for r in remaining])
        from src.chunker import Chunk
        chunks = [Chunk(chunk_id=r["chunk_id"], doc_id=r["doc_id"], text=r["text"],
                         chunk_index=r["metadata"].get("chunk_index", 0), metadata=r["metadata"])
                  for r in remaining]
        new_store.add(chunks, embeddings)
    new_store.save(config.vectorstore_dir)

    pipeline.vector_store = new_store
    pipeline.retriever.vector_store = new_store
    pipeline.retriever.refresh_bm25()

    return {"status": "deleted", "doc_id": doc_id, "remaining_chunks": str(len(remaining))}


@app.post("/reindex", response_model=UploadResponse, tags=["documents"])
def reindex() -> UploadResponse:
    """Rebuild the vector store from every document currently in `data/documents/`."""
    pipeline = get_pipeline()
    pipeline.clear_index()
    stats = pipeline.ingest_directory()
    return UploadResponse(**stats)
