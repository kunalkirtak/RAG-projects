"""
reranker.py
-----------
Lexical (BM25) indexing and rank-fusion utilities used to build hybrid
(vector + keyword) search on top of the FAISS vector store.

Rank fusion uses Reciprocal Rank Fusion (RRF), a simple, parameter-light
method that combines ranked lists from different retrieval systems without
requiring their scores to be on the same scale.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from src.logger import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """A simple BM25 lexical index over a corpus of chunk records."""

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None

    def build(self, records: List[Dict[str, Any]]) -> None:
        """(Re)build the BM25 index from a list of chunk record dicts (with a 'text' key)."""
        self._records = records
        tokenized_corpus = [_tokenize(r["text"]) for r in records]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        logger.info("BM25 index built over %d chunks", len(records))

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return the top-k records by BM25 score for `query`."""
        if self._bm25 is None or not self._records:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._records, scores), key=lambda pair: pair[1], reverse=True)
        return [{**record, "score": float(score)} for record, score in ranked[:top_k]]


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    weights: List[float] | None = None,
    key: str = "chunk_id",
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """Fuse multiple ranked result lists into a single ranking via weighted RRF.

    Each item's fused score is: sum_over_lists( weight_i / (rrf_k + rank_in_list_i) ).
    Items are deduplicated by `key`, keeping the richest record seen.

    Args:
        ranked_lists: One ranked list of record dicts per retrieval system.
        weights: Optional per-list weight (defaults to 1.0 for every list).
        key: Field used to identify duplicate records across lists.
        rrf_k: RRF damping constant (60 is the commonly used default).

    Returns:
        A single list of records sorted by fused score (descending), each
        record annotated with a `fused_score` field.
    """
    weights = weights or [1.0] * len(ranked_lists)
    fused_scores: Dict[str, float] = {}
    record_lookup: Dict[str, Dict[str, Any]] = {}

    for weight, ranked_list in zip(weights, ranked_lists):
        for rank, record in enumerate(ranked_list):
            rid = record[key]
            fused_scores[rid] = fused_scores.get(rid, 0.0) + weight / (rrf_k + rank + 1)
            record_lookup.setdefault(rid, record)

    fused = [
        {**record_lookup[rid], "fused_score": score}
        for rid, score in fused_scores.items()
    ]
    fused.sort(key=lambda r: r["fused_score"], reverse=True)
    return fused
