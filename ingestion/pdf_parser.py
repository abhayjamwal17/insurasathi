"""Page-aware PDF text extraction.

Extracts text one page at a time and preserves page-level provenance
(document_id, filename, page_number) so downstream retrieval can always
point back to the exact source page. The full document is never flattened
into a single text blob.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import fitz  # PyMuPDF

from ingestion.schemas import DocumentPage


class PDFParseError(Exception):
    """Raised when a PDF cannot be opened or parsed."""


def parse_pdf(path: str | Path, document_id: str | None = None) -> list[DocumentPage]:
    """Extract text from a PDF, one DocumentPage per page.

    Args:
        path: Path to the PDF file.
        document_id: Stable id for the document. If not provided, a new
            UUID4 is generated.

    Returns:
        A list of DocumentPage objects, one per page, in page order.

    Raises:
        PDFParseError: if the file does not exist, is not a valid PDF, or
            has zero pages.
    """
    pdf_path = Path(path)

    if not pdf_path.exists():
        raise PDFParseError(f"PDF file not found: {pdf_path}")

    if not pdf_path.is_file():
        raise PDFParseError(f"Path is not a file: {pdf_path}")

    resolved_document_id = document_id or str(uuid.uuid4())

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # PyMuPDF raises its own exception types
        raise PDFParseError(f"Failed to open PDF '{pdf_path.name}': {exc}") from exc

    try:
        if doc.page_count == 0:
            raise PDFParseError(f"PDF '{pdf_path.name}' has no pages")

        pages: list[DocumentPage] = []
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            text = page.get_text()
            pages.append(
                DocumentPage(
                    document_id=resolved_document_id,
                    filename=pdf_path.name,
                    page_number=page_index + 1,
                    text=text,
                )
            )
        return pages
    finally:
        doc.close()
