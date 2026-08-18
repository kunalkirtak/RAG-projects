"""
evaluation.py
-------------
Retrieval and pipeline evaluation utilities: Precision@K, Recall@K, latency
benchmarking, and a small harness that runs a labelled query set end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict

import numpy as np
import pandas as pd

from src.logger import get_logger
from src.utils import timer

logger = get_logger(__name__)


class LabelledQuery(TypedDict):
    """A query with a set of known-relevant chunk/doc IDs, for evaluation."""

    query: str
    relevant_ids: List[str]


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of the top-k retrieved IDs that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of all relevant IDs that appear in the top-k retrieved results."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for rid in relevant_ids if rid in top_k)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """1 / rank of the first relevant result, or 0 if none are relevant."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(pipeline: "Any", labelled_queries: List[LabelledQuery],
                        k: int = 5) -> pd.DataFrame:
    """Run retrieval-only evaluation over a labelled query set.

    Args:
        pipeline: A `RAGPipeline` instance (only `.retriever` is used).
        labelled_queries: Queries with their known-relevant chunk IDs.
        k: Cutoff for Precision@K / Recall@K.

    Returns:
        A DataFrame with one row per query and columns:
        query, precision_at_k, recall_at_k, mrr, latency_s.
    """
    rows = []
    for item in labelled_queries:
        with timer() as t:
            result = pipeline.retriever.retrieve(item["query"], top_k=k)
        retrieved_ids = [c["chunk_id"] for c in result["chunks"]]
        rows.append({
            "query": item["query"],
            "precision_at_k": precision_at_k(retrieved_ids, item["relevant_ids"], k),
            "recall_at_k": recall_at_k(retrieved_ids, item["relevant_ids"], k),
            "mrr": mean_reciprocal_rank(retrieved_ids, item["relevant_ids"]),
            "latency_s": t["elapsed_seconds"],
        })
    df = pd.DataFrame(rows)
    logger.info("Evaluated %d queries — mean P@%d=%.3f, mean R@%d=%.3f",
                len(df), k, df["precision_at_k"].mean(), k, df["recall_at_k"].mean())
    return df


def measure_latency(pipeline: "Any", queries: List[str], stream: bool = False) -> pd.DataFrame:
    """Measure end-to-end query latency (retrieval + generation) for a list of queries."""
    rows = []
    for q in queries:
        result = pipeline.query(q, use_memory=False, rewrite_query=False, stream=stream)
        timing = result["timing"]
        rows.append({
            "query": q,
            "vector_search_s": timing.get("vector_search_s", 0.0),
            "bm25_search_s": timing.get("bm25_search_s", 0.0),
            "retrieval_total_s": timing.get("total_s", 0.0),
            "pipeline_total_s": timing.get("total_pipeline_s", 0.0),
        })
    return pd.DataFrame(rows)


def summarize_latency(df: pd.DataFrame) -> Dict[str, float]:
    """Compute mean/median/p95 latency summary statistics from `measure_latency` output."""
    values = df["pipeline_total_s"].to_numpy()
    if values.size == 0:
        return {"mean_s": 0.0, "median_s": 0.0, "p95_s": 0.0}
    return {
        "mean_s": float(np.mean(values)),
        "median_s": float(np.median(values)),
        "p95_s": float(np.percentile(values, 95)),
    }
