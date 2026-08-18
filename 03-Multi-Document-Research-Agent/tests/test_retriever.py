"""Unit tests for src/retriever.py and src/reranker.py"""

from src.reranker import BM25Index, reciprocal_rank_fusion
from src.retriever import compress_context, deduplicate


def _record(chunk_id, text, **extra):
    return {"chunk_id": chunk_id, "doc_id": "doc_1", "text": text, "metadata": {}, **extra}


def test_deduplicate_removes_near_identical_chunks():
    records = [
        _record("c1", "The quick brown fox jumps over the lazy dog."),
        _record("c2", "The quick brown fox jumps over the lazy dog!"),  # near-duplicate
        _record("c3", "A completely different sentence about cats."),
    ]
    deduped = deduplicate(records)
    assert len(deduped) == 2
    assert deduped[0]["chunk_id"] == "c1"
    assert deduped[1]["chunk_id"] == "c3"


def test_compress_context_respects_char_budget():
    records = [_record("c1", "x" * 100), _record("c2", "y" * 100), _record("c3", "z" * 100)]
    compressed = compress_context(records, max_chars=150)
    total_chars = sum(len(r["text"]) for r in compressed)
    assert total_chars <= 155  # small slack for ellipsis character
    assert len(compressed) <= 2


def test_bm25_index_ranks_relevant_document_higher():
    records = [
        _record("c1", "Python is a popular programming language for data science."),
        _record("c2", "Bananas are a good source of potassium."),
    ]
    index = BM25Index()
    index.build(records)
    results = index.search("programming language", top_k=2)
    assert results[0]["chunk_id"] == "c1"


def test_reciprocal_rank_fusion_combines_rankings():
    list_a = [_record("c1", "a"), _record("c2", "b"), _record("c3", "c")]
    list_b = [_record("c3", "c"), _record("c1", "a"), _record("c2", "b")]
    fused = reciprocal_rank_fusion([list_a, list_b], weights=[0.5, 0.5])
    assert {r["chunk_id"] for r in fused} == {"c1", "c2", "c3"}
    assert fused[0]["fused_score"] >= fused[-1]["fused_score"]
