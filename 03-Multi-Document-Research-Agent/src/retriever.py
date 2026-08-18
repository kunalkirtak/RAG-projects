"""
retriever.py
------------
The `HybridRetriever` ties together vector search (FAISS) and lexical
search (BM25), fuses their rankings, removes near-duplicate chunks, applies
metadata filtering, and performs light context compression so the final
context passed to the LLM is compact and non-redundant.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from src.config import config
from src.embedder import Embedder
from src.logger import get_logger
from src.reranker import BM25Index, reciprocal_rank_fusion
from src.utils import timer
from src.vector_store import VectorStore

logger = get_logger(__name__)


def _is_near_duplicate(a: str, b: str, threshold: float = 0.9) -> bool:
    """Cheap near-duplicate detection using difflib's SequenceMatcher ratio."""
    if abs(len(a) - len(b)) / max(len(a), len(b), 1) > 0.3:
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


def deduplicate(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove exact and near-duplicate chunks, keeping the highest-scored copy."""
    kept: List[Dict[str, Any]] = []
    for record in records:
        if any(_is_near_duplicate(record["text"], k["text"]) for k in kept):
            continue
        kept.append(record)
    return kept


def compress_context(records: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    """Trim the list of retrieved records so their combined text fits `max_chars`.

    Greedily keeps the highest-ranked records first, truncating the last one
    that would overflow the budget rather than dropping it entirely.
    """
    compressed: List[Dict[str, Any]] = []
    used = 0
    for record in records:
        remaining = max_chars - used
        if remaining <= 0:
            break
        text = record["text"]
        if len(text) > remaining:
            text = text[:remaining].rsplit(" ", 1)[0] + "…"
        compressed.append({**record, "text": text})
        used += len(text)
    return compressed


class HybridRetriever:
    """Combines vector similarity search and BM25 lexical search.

    Args:
        vector_store: A populated `VectorStore`.
        embedder: The `Embedder` used to embed queries.
        alpha: Weight applied to the vector-search ranking in fusion
               (1 - alpha is applied to BM25).
    """

    def __init__(self, vector_store: VectorStore, embedder: Embedder,
                 alpha: float | None = None) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.alpha = alpha if alpha is not None else config.hybrid_alpha
        self.bm25_index = BM25Index()
        self.bm25_index.build(vector_store.records)

    def refresh_bm25(self) -> None:
        """Rebuild the BM25 index — call after `vector_store.add(...)`."""
        self.bm25_index.build(self.vector_store.records)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        max_context_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run hybrid retrieval for `query`, returning ranked, deduplicated,
        compressed chunks along with timing information.

        Returns:
            {
              "chunks": [...],       # final list of chunk records w/ scores
              "timing": {"vector_search_s": .., "bm25_search_s": .., "total_s": ..},
            }
        """
        top_k = top_k or config.top_k
        fetch_k = max(top_k * 3, 10)
        timing: Dict[str, float] = {}

        with timer() as t_total:
            with timer() as t_vec:
                query_embedding = self.embedder.embed_query(query)
                vector_results = self.vector_store.search(
                    query_embedding, top_k=fetch_k, metadata_filter=metadata_filter
                )
            timing["vector_search_s"] = t_vec["elapsed_seconds"]

            with timer() as t_bm25:
                bm25_results = self.bm25_index.search(query, top_k=fetch_k)
                if metadata_filter:
                    bm25_results = [
                        r for r in bm25_results
                        if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
                    ]
            timing["bm25_search_s"] = t_bm25["elapsed_seconds"]

            fused = reciprocal_rank_fusion(
                [vector_results, bm25_results],
                weights=[self.alpha, 1 - self.alpha],
            )
            deduped = deduplicate(fused)[:top_k]

            max_chars = max_context_chars or config.max_context_tokens * 4  # ~4 chars/token
            final_chunks = compress_context(deduped, max_chars)

        timing["total_s"] = t_total["elapsed_seconds"]
        logger.info("Retrieved %d chunks for query in %.3fs", len(final_chunks), timing["total_s"])
        return {"chunks": final_chunks, "timing": timing}


def format_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Produce a compact, citation-friendly summary of retrieved chunks."""
    sources = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        sources.append({
            "citation": f"[{i}]",
            "filename": meta.get("filename", "unknown"),
            "chunk_index": meta.get("chunk_index"),
            "score": round(chunk.get("fused_score", chunk.get("score", 0.0)), 4),
        })
    return sources
