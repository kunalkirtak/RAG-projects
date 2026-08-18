"""
embedder.py
-----------
Wraps a SentenceTransformers model to produce dense embeddings for chunks
and queries. Embeddings are L2-normalised so that inner-product search in
FAISS is equivalent to cosine similarity.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """Sentence embedding model wrapper.

    Args:
        model_name: Name of the SentenceTransformers model to load.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or config.embedding_model
        logger.info("Loading embedding model '%s'...", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        # `get_sentence_embedding_dimension` was renamed to `get_embedding_dimension`
        # in newer sentence-transformers releases; support both so this works
        # regardless of which version is installed.
        if hasattr(self._model, "get_embedding_dimension"):
            self.dimension = self._model.get_embedding_dimension()
        else:  # pragma: no cover - older sentence-transformers versions
            self.dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded (dim=%d)", self.dimension)

    def embed_texts(self, texts: List[str], batch_size: int = 32,
                     show_progress: bool = False) -> np.ndarray:
        """Embed a batch of texts, returning an (N, dim) L2-normalised float32 array."""
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string, returning a (dim,) L2-normalised float32 vector."""
        return self.embed_texts([query])[0]
