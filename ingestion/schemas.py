"""Core data structures for page-aware document ingestion.

These models exist to guarantee that every piece of extracted text can be
traced back to a specific document and page. This provenance is required
later so the RAG/evidence layer can answer "what page did this come from?".
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    """A single page of text extracted from a source PDF."""

    document_id: str = Field(..., description="Stable identifier for the source document")
    filename: str = Field(..., description="Original filename of the source PDF")
    page_number: int = Field(..., ge=1, description="1-indexed page number within the document")
    text: str = Field(..., description="Raw extracted text for this page")


class Chunk(BaseModel):
    """A chunk of text derived from one or more DocumentPage objects.

    A chunk always maps back to a single source page. If a logical chunk
    would span a page boundary, it is split so each Chunk stays attributable
    to exactly one page_number.
    """

    document_id: str
    filename: str
    page_number: int
    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    text: str
    section: str | None = Field(
        default=None,
        description="Section/heading this chunk belongs to, if reliably identified. "
        "None if no heading could be detected — never fabricated.",
    )
