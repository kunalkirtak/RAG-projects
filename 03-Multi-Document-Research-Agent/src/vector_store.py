"""
vector_store.py
----------------
FAISS-backed vector database with persistence and incremental indexing.

Design notes:
    * Uses `IndexFlatIP` (inner product) over L2-normalised vectors, which is
      mathematically equivalent to cosine similarity search.
    * Chunk text + metadata are kept in a parallel Python list (`self.records`)
      and persisted alongside the FAISS index as JSON, since FAISS itself
      only stores vectors.
    * `add()` supports incremental indexing: new chunks are appended to the
      existing index without rebuilding it from scratch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.chunker import Chunk
from src.logger import get_logger

logger = get_logger(__name__)

_INDEX_FILE = "index.faiss"
_RECORDS_FILE = "records.json"


class VectorStore:
    """A persistent, incrementally-updatable FAISS vector store."""

    def __init__(self, dimension: int) -> None:
        import faiss

        self._faiss = faiss
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.records: List[Dict[str, Any]] = []  # parallel array: records[i] <-> index vector i

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Add new chunks and their embeddings to the index (incremental)."""
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Number of chunks must match number of embedding rows")
        if embeddings.shape[0] == 0:
            return

        existing_ids = {r["chunk_id"] for r in self.records}
        new_chunks, new_vectors = [], []
        for chunk, vector in zip(chunks, embeddings):
            if chunk.chunk_id in existing_ids:
                continue  # avoid duplicate indexing on repeated ingestion
            new_chunks.append(chunk)
            new_vectors.append(vector)

        if not new_chunks:
            logger.info("No new chunks to add (all already indexed).")
            return

        vectors = np.vstack(new_vectors).astype("float32")
        self.index.add(vectors)
        for chunk in new_chunks:
            self.records.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            })
        logger.info("Added %d new vectors (total=%d)", len(new_chunks), self.index.ntotal)

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search the index for the `top_k` most similar chunks.

        Args:
            query_embedding: (dim,) L2-normalised query vector.
            top_k: Number of results to return.
            metadata_filter: Optional dict of exact-match metadata constraints,
                              e.g. {"file_type": "pdf"}.

        Returns:
            A list of result dicts with keys: chunk_id, doc_id, text, metadata, score.
        """
        if self.index.ntotal == 0:
            return []

        # Over-fetch when filtering so we still end up with `top_k` after filtering.
        fetch_k = top_k * 5 if metadata_filter else top_k
        fetch_k = min(fetch_k, self.index.ntotal)

        query_vector = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query_vector, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            record = self.records[idx]
            if metadata_filter and not _matches_filter(record["metadata"], metadata_filter):
                continue
            results.append({**record, "score": float(score)})
            if len(results) >= top_k:
                break
        return results

    def save(self, directory: str | Path) -> None:
        """Persist the FAISS index and metadata records to `directory`."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(directory / _INDEX_FILE))
        with open(directory / _RECORDS_FILE, "w", encoding="utf-8") as fh:
            json.dump({"dimension": self.dimension, "records": self.records}, fh)
        logger.info("Vector store saved to %s (%d vectors)", directory, self.index.ntotal)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """Load a previously persisted vector store from `directory`."""
        import faiss

        directory = Path(directory)
        index_path = directory / _INDEX_FILE
        records_path = directory / _RECORDS_FILE
        if not index_path.exists() or not records_path.exists():
            raise FileNotFoundError(f"No vector store found at {directory}")

        with open(records_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        store = cls(dimension=data["dimension"])
        store.index = faiss.read_index(str(index_path))
        store.records = data["records"]
        logger.info("Vector store loaded from %s (%d vectors)", directory, store.index.ntotal)
        return store

    @classmethod
    def exists(cls, directory: str | Path) -> bool:
        directory = Path(directory)
        return (directory / _INDEX_FILE).exists() and (directory / _RECORDS_FILE).exists()

    def __len__(self) -> int:
        return self.index.ntotal


def _matches_filter(metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
    return all(metadata.get(k) == v for k, v in filter_dict.items())
