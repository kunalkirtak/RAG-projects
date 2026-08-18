"""Unit tests for src/chunker.py"""

import pytest

from src.chunker import Chunker, _split_recursive, _split_semantic
from src.document_loader import Document


def test_recursive_split_respects_chunk_size():
    text = "word " * 500  # 2500 chars
    chunks = _split_recursive(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    # Allow a little slack since overlap can push chunks slightly over.
    assert all(len(c) <= 260 for c in chunks)


def test_recursive_split_short_text_returns_single_chunk():
    text = "short text"
    chunks = _split_recursive(text, chunk_size=200, chunk_overlap=20)
    assert chunks == [text]


def test_semantic_split_keeps_sentences_intact():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = _split_semantic(text, chunk_size=30, chunk_overlap=5)
    for chunk in chunks:
        assert chunk.strip().endswith((".", "!", "?"))


def test_chunker_invalid_overlap_raises():
    with pytest.raises(ValueError):
        Chunker(chunk_size=100, chunk_overlap=100)


def test_chunker_split_document_inherits_metadata():
    doc = Document(doc_id="doc_abc123", source="/tmp/fake.txt",
                    text="Sentence one. Sentence two. Sentence three.",
                    metadata={"filename": "fake.txt"})
    chunker = Chunker(chunk_size=50, chunk_overlap=5, strategy="semantic")
    chunks = chunker.split_document(doc)
    assert len(chunks) >= 1
    assert all(c.metadata["doc_id"] == "doc_abc123" for c in chunks)
    assert all(c.metadata["filename"] == "fake.txt" for c in chunks)
    assert chunks[0].chunk_id == "doc_abc123_chunk_0000"
