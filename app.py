"""Phase 1 CLI entry point: demonstrates page-aware PDF ingestion.

Usage:
    python app.py path/to/document.pdf

This does NOT run retrieval, extraction, or calculation — those are later
phases. It only proves the ingestion pipeline: PDF -> pages -> chunks.
"""

from __future__ import annotations

import sys

from config import settings
from ingestion import chunk_pages, parse_pdf, PDFParseError


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python app.py <path-to-pdf>")
        return 1

    pdf_path = sys.argv[1]

    try:
        pages = parse_pdf(pdf_path)
    except PDFParseError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Parsed {len(pages)} page(s) from '{pdf_path}'")

    chunks = chunk_pages(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    print(f"Produced {len(chunks)} chunk(s)")

    for chunk in chunks[:5]:
        preview = chunk.text[:80].replace("\n", " ")
        print(f"  [{chunk.chunk_id}] page {chunk.page_number}: {preview}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
