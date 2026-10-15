"""Data structures for the retrieval layer.

RetrievedChunk always carries the same provenance fields as ingestion.Chunk
(document_id, filename, page_number, chunk_id) plus a relevance_score, so
retrieval can never silently drop the page a fact came from.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RetrievalStatus = Literal["ok", "insufficient_evidence"]


class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever, with similarity and provenance."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    relevance_score: float = Field(
        ..., description="Cosine similarity to the query, in [-1, 1]; higher is more relevant"
    )
    section: str | None = None


class RetrievalResult(BaseModel):
    """The outcome of one retrieval query.

    status == "insufficient_evidence" means no candidate cleared the
    relevance threshold. In that case `chunks` is empty and callers must
    not fabricate an answer from it.
    """

    query: str
    status: RetrievalStatus
    chunks: list[RetrievedChunk] = Field(default_factory=list)
