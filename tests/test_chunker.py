from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.chunker import chunk_pages
from ingestion.pdf_parser import parse_pdf
from ingestion.schemas import DocumentPage


def test_chunking_preserves_page_provenance(synthetic_pdf: Path) -> None:
    pages = parse_pdf(synthetic_pdf, document_id="doc-abc")
    chunks = chunk_pages(pages)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.document_id == "doc-abc"
        assert chunk.filename == "synthetic_test.pdf"
        assert chunk.page_number in (1, 2, 3)


def test_every_page_produces_at_least_one_chunk(synthetic_pdf: Path) -> None:
    pages = parse_pdf(synthetic_pdf)
    chunks = chunk_pages(pages)

    page_numbers_with_chunks = {c.page_number for c in chunks}
    assert page_numbers_with_chunks == {1, 2, 3}


def test_chunk_ids_are_unique(synthetic_pdf: Path) -> None:
    pages = parse_pdf(synthetic_pdf)
    chunks = chunk_pages(pages)

    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_large_page_is_split_into_multiple_overlapping_chunks(single_page_pdf: Path) -> None:
    pages = parse_pdf(single_page_pdf)
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(c.page_number == 1 for c in chunks)
    # No chunk should wildly exceed the requested size.
    assert all(len(c.text) <= 200 + 60 for c in chunks)


def test_small_page_produces_single_chunk(synthetic_pdf: Path) -> None:
    pages = parse_pdf(synthetic_pdf)
    chunks = chunk_pages(pages, chunk_size=1200, chunk_overlap=200)

    for page_number in (1, 2, 3):
        page_chunks = [c for c in chunks if c.page_number == page_number]
        assert len(page_chunks) == 1


def test_chunking_never_crosses_page_boundary(synthetic_pdf: Path) -> None:
    pages = parse_pdf(synthetic_pdf)
    chunks = chunk_pages(pages, chunk_size=30, chunk_overlap=5)

    for chunk in chunks:
        source_page = next(p for p in pages if p.page_number == chunk.page_number)
        normalized_chunk = " ".join(chunk.text.split())
        normalized_page = " ".join(source_page.text.split())
        assert normalized_chunk in normalized_page


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    page = DocumentPage(document_id="d", filename="f.pdf", page_number=1, text="hello world")
    with pytest.raises(ValueError):
        chunk_pages([page], chunk_size=100, chunk_overlap=100)


def test_empty_page_text_produces_no_chunks() -> None:
    page = DocumentPage(document_id="d", filename="f.pdf", page_number=1, text="   ")
    chunks = chunk_pages([page])
    assert chunks == []
