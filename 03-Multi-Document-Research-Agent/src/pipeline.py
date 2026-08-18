"""
pipeline.py
-----------
`RAGPipeline` is the top-level orchestrator that wires together document
ingestion, chunking, embedding, hybrid retrieval, prompt construction, and
Gemini generation. This is the single entry point used by the CLI, the
FastAPI service, and the Streamlit app.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.chunker import Chunker
from src.config import AppConfig, config as default_config
from src.document_loader import discover_documents, load_documents
from src.embedder import Embedder
from src.gemini_client import GeminiClient
from src.logger import get_logger
from src.memory import ConversationMemory
from src.prompts import build_query_rewrite_prompt, build_rag_prompt
from src.retriever import HybridRetriever, format_sources
from src.utils import timer
from src.vector_store import VectorStore

logger = get_logger(__name__)


class RAGPipeline:
    """End-to-end multi-document RAG pipeline.

    Args:
        cfg: An `AppConfig`. Defaults to the module-level singleton config.
        lazy_gemini: If True (default), the Gemini client is only created on
                     first use — useful for ingestion-only workflows / tests
                     that don't need a valid API key.
    """

    def __init__(self, cfg: Optional[AppConfig] = None, lazy_gemini: bool = True) -> None:
        self.config = cfg or default_config
        self.chunker = Chunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            strategy=self.config.chunking_strategy,
        )
        self.embedder = Embedder(self.config.embedding_model)
        self.memory = ConversationMemory()

        if VectorStore.exists(self.config.vectorstore_dir):
            self.vector_store = VectorStore.load(self.config.vectorstore_dir)
        else:
            self.vector_store = VectorStore(dimension=self.embedder.dimension)

        self.retriever = HybridRetriever(self.vector_store, self.embedder, alpha=self.config.hybrid_alpha)

        self._gemini: Optional[GeminiClient] = None if lazy_gemini else self._build_gemini()

    def _build_gemini(self) -> GeminiClient:
        return GeminiClient(
            api_key=self.config.gemini_api_key,
            model_name=self.config.gemini_model,
            temperature=self.config.temperature,
        )

    @property
    def gemini(self) -> GeminiClient:
        if self._gemini is None:
            self._gemini = self._build_gemini()
        return self._gemini

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest_paths(self, paths: List[str]) -> Dict[str, Any]:
        """Ingest a specific list of file paths into the vector store."""
        with timer() as t:
            documents = load_documents(paths)
            chunks = self.chunker.split_documents(documents)
            embeddings = self.embedder.embed_texts([c.text for c in chunks], show_progress=True)
            self.vector_store.add(chunks, embeddings)
            self.retriever.refresh_bm25()
            self.vector_store.save(self.config.vectorstore_dir)
        logger.info("Ingested %d documents / %d chunks in %.2fs",
                    len(documents), len(chunks), t["elapsed_seconds"])
        return {
            "num_documents": len(documents),
            "num_chunks": len(chunks),
            "elapsed_seconds": t["elapsed_seconds"],
            "total_vectors": len(self.vector_store),
        }

    def ingest_directory(self, directory: Optional[str] = None) -> Dict[str, Any]:
        """Discover and ingest every supported document under `directory`."""
        directory = directory or self.config.documents_dir
        paths = [str(p) for p in discover_documents(directory)]
        if not paths:
            logger.warning("No supported documents found in %s", directory)
            return {"num_documents": 0, "num_chunks": 0, "elapsed_seconds": 0.0, "total_vectors": len(self.vector_store)}
        return self.ingest_paths(paths)

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #
    def _rewrite_query(self, query: str) -> str:
        """Resolve follow-up references in `query` using conversation history."""
        history = self.memory.format_for_prompt(max_turns=3)
        if not history:
            return query
        try:
            prompt = build_query_rewrite_prompt(query, history)
            rewritten = self.gemini.generate(prompt)
            return rewritten or query
        except Exception as exc:  # noqa: BLE001 - fall back to original query
            logger.warning("Query rewriting failed, using original query: %s", exc)
            return query

    def query(
        self,
        user_query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        use_memory: bool = True,
        rewrite_query: Optional[bool] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Answer a user question via retrieval-augmented generation.

        Returns a dict with: answer (str, or a generator if stream=True),
        sources, retrieved_chunks, rewritten_query, and timing.
        """
        with timer() as t_total:
            do_rewrite = self.config.enable_query_rewrite if rewrite_query is None else rewrite_query
            search_query = self._rewrite_query(user_query) if (do_rewrite and use_memory) else user_query
            retrieval = self.retriever.retrieve(search_query, top_k=top_k, metadata_filter=metadata_filter)
            chunks = retrieval["chunks"]
            history = self.memory.format_for_prompt() if use_memory else ""
            prompt = build_rag_prompt(user_query, chunks, history=history)

            if stream:
                answer: Any = self.gemini.generate_stream(prompt)
            else:
                if not chunks:
                    answer = (
                        "I couldn't find relevant information in the indexed documents "
                        "to answer that question."
                    )
                else:
                    try:
                        answer = self.gemini.generate(prompt)
                    except Exception as exc:  # noqa: BLE001 - surface a friendly message, not a 500
                        logger.error("Generation failed: %s", exc)
                        answer = (
                            "Sorry, I couldn't generate an answer right now — the language "
                            "model request failed. Check that GEMINI_API_KEY is valid and "
                            "try again."
                        )
                if use_memory:
                    self.memory.add_turn(user_query, answer, sources=format_sources(chunks))

        return {
            "query": user_query,
            "rewritten_query": search_query,
            "answer": answer,
            "sources": format_sources(chunks),
            "retrieved_chunks": chunks,
            "timing": {**retrieval["timing"], "total_pipeline_s": t_total["elapsed_seconds"]},
        }

    def reset_memory(self) -> None:
        self.memory.clear()

    def clear_index(self) -> None:
        """Wipe the vector store (used by the 'Clear Database' UI action)."""
        self.vector_store = VectorStore(dimension=self.embedder.dimension)
        self.retriever = HybridRetriever(self.vector_store, self.embedder, alpha=self.config.hybrid_alpha)
        self.vector_store.save(self.config.vectorstore_dir)
        logger.info("Vector store cleared.")
