"""
chunker.py
----------
Splits loaded documents into overlapping chunks suitable for embedding.

Two strategies are supported:
    * "recursive" - splits on a hierarchy of separators (paragraphs, then
      sentences, then words) trying to keep chunks close to `chunk_size`
      characters, with `chunk_overlap` characters shared between neighbours.
    * "semantic"  - splits on sentence boundaries first, then greedily packs
      whole sentences into chunks until the target size is reached, which
      keeps sentences intact (never splits mid-sentence).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.document_loader import Document
from src.logger import get_logger
from src.utils import generate_chunk_id

logger = get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """A single chunk of text ready for embedding."""

    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def _split_recursive_raw(text: str, chunk_size: int,
                          separators: List[str] | None = None) -> List[str]:
    """Recursively split text into non-overlapping pieces using a hierarchy of
    separators. Falls back to a hard character split if no separator produces
    small enough pieces (mirrors the behaviour of LangChain's
    RecursiveCharacterTextSplitter, reimplemented here to avoid a hard
    dependency).

    Overlap is intentionally NOT applied here — it is applied exactly once,
    by the top-level `_split_recursive` wrapper, to avoid compounding overlap
    at every level of recursion.
    """
    separators = separators or _SEPARATORS
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = separators[0]
    remaining_separators = separators[1:]

    if separator == "":
        pieces = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    else:
        pieces = text.split(separator)

    chunks: List[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{separator}{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(piece) > chunk_size and remaining_separators:
                chunks.extend(_split_recursive_raw(piece, chunk_size, remaining_separators))
                current = ""
            else:
                current = piece

    if current.strip():
        chunks.append(current)

    return chunks


def _split_recursive(text: str, chunk_size: int, chunk_overlap: int,
                      separators: List[str] | None = None) -> List[str]:
    """Split `text` into overlapping chunks: structural splitting via
    `_split_recursive_raw`, followed by a single pass of overlap injection.
    """
    raw_chunks = _split_recursive_raw(text, chunk_size, separators)
    return _apply_overlap(raw_chunks, chunk_overlap)


def _apply_overlap(chunks: List[str], chunk_overlap: int) -> List[str]:
    """Prepend a trailing slice of the previous chunk to create overlap."""
    if chunk_overlap <= 0 or len(chunks) < 2:
        return chunks
    overlapped = [chunks[0]]
    for prev, curr in zip(chunks, chunks[1:]):
        tail = prev[-chunk_overlap:]
        overlapped.append(f"{tail} {curr}".strip())
    return overlapped


def _split_semantic(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Pack whole sentences into chunks, never splitting mid-sentence."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_len = 0

    for sentence in sentences:
        projected = current_len + len(sentence) + 1
        if projected > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))
            # start next chunk with overlap: carry over trailing sentences
            overlap_sentences: List[str] = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)
            current_sentences = overlap_sentences
            current_len = overlap_len

        current_sentences.append(sentence)
        current_len += len(sentence) + 1

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


class Chunker:
    """Configurable document chunker.

    Args:
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap (in characters) between consecutive chunks.
        strategy: "recursive" or "semantic".
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120,
                 strategy: str = "recursive") -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if strategy not in {"recursive", "semantic"}:
            raise ValueError("strategy must be 'recursive' or 'semantic'")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    def split_text(self, text: str) -> List[str]:
        """Split a raw string into a list of chunk strings using the configured strategy."""
        if self.strategy == "semantic":
            return _split_semantic(text, self.chunk_size, self.chunk_overlap)
        return _split_recursive(text, self.chunk_size, self.chunk_overlap)

    def split_document(self, document: Document) -> List[Chunk]:
        """Split a single `Document` into a list of `Chunk` objects with inherited metadata."""
        pieces = self.split_text(document.text)
        chunks = []
        for idx, piece in enumerate(pieces):
            chunk_id = generate_chunk_id(document.doc_id, idx)
            metadata = {
                **document.metadata,
                "doc_id": document.doc_id,
                "source": document.source,
                "chunk_index": idx,
                "num_chunks": len(pieces),
            }
            chunks.append(Chunk(chunk_id=chunk_id, doc_id=document.doc_id, text=piece,
                                 chunk_index=idx, metadata=metadata))
        logger.info("Split document '%s' into %d chunks (%s strategy)",
                    document.metadata.get("filename", document.doc_id), len(chunks), self.strategy)
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """Split multiple documents, returning a flat list of chunks."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        return all_chunks
