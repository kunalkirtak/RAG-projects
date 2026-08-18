"""
utils.py
--------
Small, dependency-light helper utilities shared across the project:
text cleaning, hashing, timing, and filesystem helpers.
"""

from __future__ import annotations

import hashlib
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it doesn't already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_text(text: str) -> str:
    """Normalise whitespace and strip control characters from raw extracted text.

    - Collapses runs of whitespace into single spaces (but preserves paragraph
      breaks as double newlines).
    - Removes null bytes and other non-printable control characters.
    """
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def stable_hash(text: str, length: int = 12) -> str:
    """Return a short, stable hex hash for a piece of text (used for IDs)."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length]


def generate_doc_id(source_path: str) -> str:
    """Generate a stable document ID derived from its source path."""
    return f"doc_{stable_hash(source_path)}"


def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate a deterministic chunk ID from a document ID and chunk position."""
    return f"{doc_id}_chunk_{chunk_index:04d}"


@contextmanager
def timer() -> Iterator[dict]:
    """Context manager that measures elapsed wall-clock time.

    Usage:
        with timer() as t:
            do_work()
        print(t["elapsed_seconds"])
    """
    state = {"elapsed_seconds": 0.0}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["elapsed_seconds"] = time.perf_counter() - start


def truncate(text: str, max_chars: int = 200) -> str:
    """Truncate text for display purposes, appending an ellipsis if needed."""
    text = text.strip()
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"
