"""Configuration management for the ICU / Hospital Interim Bill Auditor.

Loads settings from environment variables (via a local .env file, if present).
No secrets or API keys are hardcoded here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    policies_dir: Path
    bills_dir: Path
    chunk_size: int
    chunk_overlap: int
    embedding_model_name: str
    vector_store_dir: Path
    retrieval_top_k: int
    retrieval_relevance_threshold: float
    llm_provider: str
    groq_api_key: str | None
    groq_model: str
    ollama_base_url: str
    ollama_model: str


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def load_settings() -> Settings:
    """Build a Settings object from environment variables, with sane defaults."""
    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    policies_dir = Path(os.getenv("POLICIES_DIR", str(data_dir / "policies")))
    bills_dir = Path(os.getenv("BILLS_DIR", str(data_dir / "bills")))
    chunk_size = _get_int("CHUNK_SIZE", 1200)
    chunk_overlap = _get_int("CHUNK_OVERLAP", 200)

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"CHUNK_OVERLAP ({chunk_overlap}) must be smaller than CHUNK_SIZE ({chunk_size})"
        )

    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    vector_store_dir = Path(os.getenv("VECTOR_STORE_DIR", str(data_dir / "vector_store")))
    retrieval_top_k = _get_int("RETRIEVAL_TOP_K", 5)
    retrieval_relevance_threshold = _get_float("RETRIEVAL_RELEVANCE_THRESHOLD", 0.35)

    llm_provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    groq_api_key = os.getenv("GROQ_API_KEY") or None
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")

    return Settings(
        data_dir=data_dir,
        policies_dir=policies_dir,
        bills_dir=bills_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model_name=embedding_model_name,
        vector_store_dir=vector_store_dir,
        retrieval_top_k=retrieval_top_k,
        retrieval_relevance_threshold=retrieval_relevance_threshold,
        llm_provider=llm_provider,
        groq_api_key=groq_api_key,
        groq_model=groq_model,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
    )


settings = load_settings()
