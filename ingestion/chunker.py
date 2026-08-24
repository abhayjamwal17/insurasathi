"""Deterministic, page-preserving text chunking.

Chunking never merges text across a page boundary — each Chunk stays
attributable to exactly one source page_number. Chunking is character-based
(not token-based) for Phase 1 simplicity and transparency.

No section/heading is invented: `section` stays None unless a simple,
reliable heuristic (a short, standalone, capitalized line) identifies one.
"""

from __future__ import annotations

from ingestion.schemas import Chunk, DocumentPage

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 50


def _detect_heading(text: str) -> str | None:
    """Best-effort heading detection for the first line of a chunk.

    Only returns a heading when confident: a short, standalone line that is
    fully uppercase or title-cased, with no trailing sentence punctuation.
    Returns None otherwise rather than guessing.
    """
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not first_line:
        return None
    if len(first_line) > 80:
        return None
    if first_line.endswith((".", ",", ";")):
        return None
    words = first_line.split()
    if not words:
        return None
    if first_line.isupper() or first_line.istitle():
        return first_line
    return None


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping windows, trying to break on whitespace."""
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    spans: list[tuple[int, int]] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            break_point = text.rfind(" ", start, end)
            if break_point != -1 and break_point > start:
                end = break_point

        spans.append((start, end))

        if end >= text_len:
            break

        next_start = end - chunk_overlap
        start = next_start if next_start > start else end

    # Avoid a tiny trailing chunk: extend the previous span to cover it
    # instead of appending a small final piece (no duplicated overlap text).
    if len(spans) > 1 and (spans[-1][1] - spans[-1][0]) < MIN_CHUNK_SIZE:
        (prev_start, _prev_end), (_last_start, last_end) = spans[-2], spans[-1]
        spans[-2] = (prev_start, last_end)
        spans.pop()

    return [text[s:e].strip() for s, e in spans if text[s:e].strip()]


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk a list of DocumentPage objects, preserving page provenance.

    Args:
        pages: Pages to chunk, typically the output of parse_pdf.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive
            chunks within the same page.

    Returns:
        A list of Chunk objects. Every chunk's page_number, document_id and
        filename match the page it was derived from.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    chunks: list[Chunk] = []
    for page in pages:
        pieces = _split_text(page.text, chunk_size, chunk_overlap)
        for piece_index, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    document_id=page.document_id,
                    filename=page.filename,
                    page_number=page.page_number,
                    chunk_id=f"{page.document_id}_p{page.page_number}_c{piece_index}",
                    text=piece,
                    section=_detect_heading(piece),
                )
            )

    return chunks
