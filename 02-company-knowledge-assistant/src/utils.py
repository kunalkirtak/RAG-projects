"""
utils.py
========
Shared utilities: logging configuration, environment configuration, and the
Gemini API client wrapper.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import google.generativeai as genai


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for the whole application."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return  # Already configured.

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    """Application configuration, sourced from environment variables."""

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    score_threshold: float = float(os.getenv("SCORE_THRESHOLD", "0.0"))
    memory_limit: int = int(os.getenv("MEMORY_LIMIT", "10"))
    vector_store_dir: str = os.getenv("VECTOR_STORE_DIR", "data/vector_store")
    upload_dir: str = os.getenv("UPLOAD_DIR", "data/uploaded_docs")


settings = Settings()


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

class GeminiError(Exception):
    """Raised when the Gemini API cannot be reached or returns an error."""


class GeminiClient:
    """
    Thin wrapper around google-generativeai that adds retries, logging,
    and clear error messages for the RAG pipeline.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.gemini_model
        self.max_retries = max_retries
        self._model = None

        if not self.api_key or self.api_key == "PASTE_YOUR_API_KEY_HERE":
            logger.warning("GEMINI_API_KEY is not set. Set it before making generation calls.")

    def _get_model(self):
        if self._model is None:
            try:
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(self.model_name)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to initialize Gemini model.")
                raise GeminiError(
                    f"Failed to initialize Gemini model '{self.model_name}': {exc}"
                ) from exc
        return self._model

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Generate a response from Gemini for the given prompt, with retries."""
        if not self.api_key or self.api_key == "PASTE_YOUR_API_KEY_HERE":
            raise GeminiError(
                "GEMINI_API_KEY is missing or is still the placeholder value. "
                "Set your real Gemini API key before chatting."
            )

        model = self._get_model()
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 2):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": temperature},
                )
                text = getattr(response, "text", None)
                if not text:
                    raise GeminiError("Gemini returned an empty response.")
                return text.strip()
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s",
                    attempt, self.max_retries + 1, exc,
                )
                time.sleep(min(2 ** attempt, 8))

        logger.error("Gemini call failed after retries.")
        raise GeminiError(
            f"Gemini API call failed after {self.max_retries + 1} attempts: {last_exception}"
        )
