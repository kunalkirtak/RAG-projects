"""
config.py
---------
Central configuration system for the Multi-Document Research Agent.

Configuration is resolved in the following priority order (highest wins):
    1. Explicit keyword arguments passed to `AppConfig(...)`
    2. Environment variables (loaded from `.env` via python-dotenv)
    3. `configs/config.yaml` file values
    4. Hard-coded defaults defined below

The system is intentionally dependency-light: YAML is optional, and the
whole module degrades gracefully if `configs/config.yaml` is missing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a soft dependency
    pass

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a soft dependency
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load a YAML config file if it exists, returning an empty dict otherwise."""
    if yaml is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


@dataclass
class AppConfig:
    """Strongly-typed application configuration.

    Attributes:
        gemini_api_key: API key used to authenticate against the Gemini API.
        gemini_model: Model name used for generation (e.g. "gemini-3.5-flash").
        embedding_model: SentenceTransformers model used to embed text.
        chunk_size: Target number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        chunking_strategy: Either "recursive" or "semantic".
        top_k: Default number of chunks to retrieve per query.
        hybrid_alpha: Weight given to vector search vs BM25 in [0, 1].
                      1.0 = pure vector search, 0.0 = pure BM25.
        max_context_tokens: Maximum number of tokens allowed in the assembled
                             context that is sent to the LLM.
        temperature: Sampling temperature for generation.
        vectorstore_dir: Directory where the FAISS index + metadata are stored.
        documents_dir: Directory where uploaded/source documents live.
        logs_dir: Directory where log files are written.
        request_timeout: Timeout (seconds) for outbound Gemini API calls.
        max_retries: Number of retry attempts for transient API failures.
    """

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size: int = 800
    chunk_overlap: int = 120
    chunking_strategy: str = "recursive"  # "recursive" | "semantic"

    top_k: int = 5
    hybrid_alpha: float = 0.6
    max_context_tokens: int = 3000
    temperature: float = 0.3

    vectorstore_dir: str = str(PROJECT_ROOT / "data" / "vectorstore")
    documents_dir: str = str(PROJECT_ROOT / "data" / "documents")
    logs_dir: str = str(PROJECT_ROOT / "logs")

    request_timeout: int = 60
    max_retries: int = 3

    # Free-tier friendliness: a free Gemini key allows only a handful of
    # requests per minute. min_request_interval_s spaces calls out so we do
    # not trip a 429, and enable_query_rewrite is off by default since it
    # costs a whole extra Gemini call per turn.
    min_request_interval_s: float = 4.0
    enable_query_rewrite: bool = False

    def __post_init__(self) -> None:
        Path(self.vectorstore_dir).mkdir(parents=True, exist_ok=True)
        Path(self.documents_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict, masking the API key."""
        data = asdict(self)
        if data.get("gemini_api_key"):
            data["gemini_api_key"] = "***REDACTED***"
        return data


def load_config(yaml_path: Optional[str] = None, **overrides: Any) -> AppConfig:
    """Build an `AppConfig`, merging YAML file, environment variables, and overrides.

    Args:
        yaml_path: Optional explicit path to a YAML config file. Defaults to
                   `configs/config.yaml` relative to the project root.
        **overrides: Explicit keyword overrides, take highest priority.

    Returns:
        A fully resolved `AppConfig` instance.
    """
    yaml_file = Path(yaml_path) if yaml_path else PROJECT_ROOT / "configs" / "config.yaml"
    yaml_values = _load_yaml_config(yaml_file)

    env_values: Dict[str, Any] = {}
    env_map = {
        "GEMINI_API_KEY": "gemini_api_key",
        "GEMINI_MODEL": "gemini_model",
        "EMBEDDING_MODEL": "embedding_model",
        "CHUNK_SIZE": ("chunk_size", int),
        "CHUNK_OVERLAP": ("chunk_overlap", int),
        "CHUNKING_STRATEGY": "chunking_strategy",
        "TOP_K": ("top_k", int),
        "HYBRID_ALPHA": ("hybrid_alpha", float),
        "MAX_CONTEXT_TOKENS": ("max_context_tokens", int),
        "MIN_REQUEST_INTERVAL_S": ("min_request_interval_s", float),
        "ENABLE_QUERY_REWRITE": ("enable_query_rewrite", lambda v: v.lower() in ("1", "true", "yes")),
        "TEMPERATURE": ("temperature", float),
        "VECTORSTORE_DIR": "vectorstore_dir",
        "DOCUMENTS_DIR": "documents_dir",
        "LOGS_DIR": "logs_dir",
    }
    for env_key, target in env_map.items():
        raw = os.getenv(env_key)
        if raw is None:
            continue
        if isinstance(target, tuple):
            field_name, caster = target
            env_values[field_name] = caster(raw)
        else:
            env_values[target] = raw

    merged = {**yaml_values, **env_values, **overrides}
    valid_fields = {f for f in AppConfig.__dataclass_fields__}
    merged = {k: v for k, v in merged.items() if k in valid_fields}
    return AppConfig(**merged)


# Module-level singleton, lazily created the first time it's imported by the app.
config = load_config()
