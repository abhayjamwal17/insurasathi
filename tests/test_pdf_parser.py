from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.pdf_parser import PDFParseError, parse_pdf


def test_pdf_loads_successfully(synthetic_pdf: Path) -> None:
    pages = parse_pdf(synthetic_pdf)
    assert len(pages) > 0


def test_page_count_is_correct(synthetic_pdf: Path) -> None:
    pages = parse_pdf(synthetic_pdf)
    assert len(pages) == 3


def test_text_is_extracted(synthetic_pdf: Path) -> None:
    pages = parse_pdf(synthetic_pdf)
    assert "synthetic test content for page one" in pages[0].text
    assert "synthetic test content for page two" in pages[1].text
    assert "synthetic test content for page three" in pages[2].text


def test_page_numbers_are_preserved_and_ordered(synthetic_pdf: Path) -> None:
    pages = parse_pdf(synthetic_pdf)
    assert [p.page_number for p in pages] == [1, 2, 3]


def test_metadata_is_preserved(synthetic_pdf: Path) -> None:
    pages = parse_pdf(synthetic_pdf, document_id="doc-123")
    for page in pages:
        assert page.document_id == "doc-123"
        assert page.filename == "synthetic_test.pdf"


def test_document_id_is_generated_when_not_provided(synthetic_pdf: Path) -> None:
    pages = parse_pdf(synthetic_pdf)
    assert pages[0].document_id
    assert all(p.document_id == pages[0].document_id for p in pages)


def test_missing_file_raises_pdf_parse_error(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(PDFParseError):
        parse_pdf(missing)


def test_invalid_pdf_raises_pdf_parse_error(invalid_pdf: Path) -> None:
    with pytest.raises(PDFParseError):
        parse_pdf(invalid_pdf)


def test_empty_pdf_raises_pdf_parse_error(empty_pdf: Path) -> None:
    with pytest.raises(PDFParseError):
        parse_pdf(empty_pdf)


def test_directory_path_raises_pdf_parse_error(tmp_path: Path) -> None:
    with pytest.raises(PDFParseError):
        parse_pdf(tmp_path)
