"""
gemini_client.py
----------------
Thin, resilient wrapper around the `google-genai` SDK (the officially
supported, unified SDK that replaced the deprecated `google-generativeai`
package). Centralises API-key configuration, retry/backoff on transient
errors, and exposes both blocking and streaming generation.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


class GeminiClientError(Exception):
    """Raised when the Gemini API call ultimately fails after retries."""


class GeminiClient:
    """Wraps `google.genai` for text generation.

    Args:
        api_key: Gemini API key. Falls back to `config.gemini_api_key`.
        model_name: Model to use, e.g. "gemini-3.5-flash".
        temperature: Sampling temperature.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None,
                 temperature: Optional[float] = None) -> None:
        from google import genai

        self.api_key = api_key or config.gemini_api_key
        if not self.api_key or self.api_key == "PASTE_YOUR_API_KEY_HERE":
            logger.warning(
                "No valid Gemini API key configured. Set GEMINI_API_KEY before generating."
            )

        self._genai = genai
        self._client = genai.Client(api_key=self.api_key)
        self.model_name = model_name or config.gemini_model
        self.temperature = temperature if temperature is not None else config.temperature
        self.min_interval = config.min_request_interval_s
        self._last_call_ts = 0.0

    def _throttle(self) -> None:
        """Sleep just long enough to respect min_request_interval_s between calls.

        Free-tier Gemini keys are capped at a handful of requests per minute,
        so a naive loop of back-to-back calls trips a 429 almost immediately.
        """
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call_ts = time.monotonic()

    def generate(self, prompt: str, max_retries: Optional[int] = None) -> str:
        """Generate a complete (non-streaming) response for `prompt`.

        Retries transient failures with exponential backoff.
        """
        from google.genai import types

        max_retries = max_retries if max_retries is not None else config.max_retries
        last_error: Exception | None = None
        self._throttle()

        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=self.temperature),
                )
                return (response.text or "").strip()
            except Exception as exc:  # noqa: BLE001 - broad by design, then re-raised
                last_error = exc
                wait = min(2 ** attempt, 20)
                logger.warning("Gemini generate() attempt %d/%d failed: %s. Retrying in %ds",
                               attempt, max_retries, exc, wait)
                time.sleep(wait)

        raise GeminiClientError(f"Gemini generation failed after {max_retries} attempts: {last_error}")

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Yield response text incrementally as it is generated.

        On failure, yields a single error message rather than raising, so
        streaming consumers (CLI/Streamlit) can display it gracefully.
        """
        from google.genai import types

        try:
            self._throttle()
            stream = self._client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=self.temperature),
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:  # noqa: BLE001
            logger.error("Gemini streaming generation failed: %s", exc)
            yield f"\n[Error: generation failed — {exc}]"
