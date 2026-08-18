"""
visualization.py
-----------------
Plotting helpers for embedding space visualisation and retrieval timing,
built on Plotly so they render nicely both in Colab/Jupyter and when saved
as standalone HTML in the Streamlit app.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from src.logger import get_logger

logger = get_logger(__name__)


def plot_embedding_space(embeddings: np.ndarray, labels: List[str],
                          color_by: Optional[List[str]] = None,
                          title: str = "Document Chunk Embedding Space (PCA)") -> go.Figure:
    """Project high-dimensional chunk embeddings into 2D with PCA and scatter-plot them.

    Args:
        embeddings: (N, dim) array of chunk embeddings.
        labels: Hover labels (e.g. truncated chunk text) for each point.
        color_by: Optional categorical value per point (e.g. source filename)
                  used to colour the scatter plot.
        title: Chart title.

    Returns:
        A Plotly `Figure`.
    """
    if embeddings.shape[0] < 2:
        logger.warning("Need at least 2 embeddings to plot; got %d", embeddings.shape[0])
        return go.Figure()

    n_components = 2
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(embeddings)

    df = pd.DataFrame({
        "x": projected[:, 0],
        "y": projected[:, 1],
        "label": labels,
        "source": color_by if color_by else ["all"] * len(labels),
    })
    fig = px.scatter(
        df, x="x", y="y", color="source", hover_name="label",
        title=title, template="plotly_dark",
    )
    fig.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=1, color="white")))
    return fig


def plot_retrieval_timing(timing_df: pd.DataFrame,
                           title: str = "Retrieval & Generation Latency by Query") -> go.Figure:
    """Bar chart comparing vector search / BM25 / total latency across evaluated queries."""
    fig = go.Figure()
    if "vector_search_s" in timing_df:
        fig.add_bar(name="Vector search", x=timing_df.index, y=timing_df["vector_search_s"])
    if "bm25_search_s" in timing_df:
        fig.add_bar(name="BM25 search", x=timing_df.index, y=timing_df["bm25_search_s"])
    if "pipeline_total_s" in timing_df:
        fig.add_bar(name="Total pipeline", x=timing_df.index, y=timing_df["pipeline_total_s"])
    fig.update_layout(barmode="group", title=title, template="plotly_dark",
                       xaxis_title="Query index", yaxis_title="Seconds")
    return fig


def plot_precision_recall(eval_df: pd.DataFrame,
                           title: str = "Precision@K / Recall@K by Query") -> go.Figure:
    """Bar chart of per-query Precision@K and Recall@K from `evaluate_retrieval` output."""
    fig = go.Figure()
    fig.add_bar(name="Precision@K", x=eval_df["query"], y=eval_df["precision_at_k"])
    fig.add_bar(name="Recall@K", x=eval_df["query"], y=eval_df["recall_at_k"])
    fig.update_layout(barmode="group", title=title, template="plotly_dark",
                       xaxis_title="Query", yaxis_title="Score", xaxis_tickangle=-30)
    return fig
