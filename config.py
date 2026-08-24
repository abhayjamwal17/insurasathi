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


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


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

    return Settings(
        data_dir=data_dir,
        policies_dir=policies_dir,
        bills_dir=bills_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


settings = load_settings()
