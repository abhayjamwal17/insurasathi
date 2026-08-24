"""Shared pytest fixtures: synthetic PDFs for unit testing only.

These PDFs contain placeholder text (e.g. "Page 1 content") for structural
testing. They are NOT real insurance policy content and must never be
presented as such.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest


def _make_pdf(path: Path, page_texts: list[str]) -> Path:
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        # insert_text does not wrap long text onto multiple lines; use a
        # textbox so long fixture strings are fully rendered onto the page.
        page.insert_textbox(fitz.Rect(72, 72, 500, 750), text)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def synthetic_pdf(tmp_path: Path) -> Path:
    """A 3-page synthetic PDF with distinct, identifiable text per page."""
    pdf_path = tmp_path / "synthetic_test.pdf"
    return _make_pdf(
        pdf_path,
        [
            "SECTION ONE\nThis is synthetic test content for page one.",
            "SECTION TWO\nThis is synthetic test content for page two.",
            "SECTION THREE\nThis is synthetic test content for page three.",
        ],
    )


@pytest.fixture
def single_page_pdf(tmp_path: Path) -> Path:
    """A 1-page synthetic PDF with enough text to force multiple chunks."""
    pdf_path = tmp_path / "single_page.pdf"
    long_text = "This is a repeated sentence used only for chunking tests. " * 60
    return _make_pdf(pdf_path, [long_text])


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    """A zero-byte file with a .pdf extension (not a valid PDF)."""
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.touch()
    return pdf_path


@pytest.fixture
def invalid_pdf(tmp_path: Path) -> Path:
    """A file with a .pdf extension that is not actually a valid PDF."""
    pdf_path = tmp_path / "not_a_real.pdf"
    pdf_path.write_text("This is plain text, not a PDF file.")
    return pdf_path
