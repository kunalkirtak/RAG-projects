"""Unit tests for src/document_loader.py"""

import pytest

from src.document_loader import DocumentLoadError, load_document
from src.utils import clean_text


def test_clean_text_collapses_whitespace():
    dirty = "Hello   world\n\n\n\nGoodbye\x00 world"
    cleaned = clean_text(dirty)
    assert "\x00" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Hello world" in cleaned


def test_load_txt_document(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a   test document.\n\n\nWith extra whitespace.")

    document = load_document(file_path)

    assert document.source == str(file_path)
    assert "test document" in document.text
    assert document.metadata["file_type"] == "txt"
    assert document.metadata["num_words"] > 0
    assert document.doc_id.startswith("doc_")


def test_load_document_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(DocumentLoadError):
        load_document(missing)


def test_load_document_unsupported_extension_raises(tmp_path):
    bad_file = tmp_path / "sample.xyz"
    bad_file.write_text("data")
    with pytest.raises(DocumentLoadError):
        load_document(bad_file)
