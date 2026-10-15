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


# ---------------------------------------------------------------------------
# Phase 2 (RAG retrieval) fixtures
#
# SYNTHETIC_POLICY_PAGES below is clearly-labeled, invented test content used
# only to exercise the retrieval pipeline end to end. It is NOT a real
# insurance policy and must never be presented as one.
# ---------------------------------------------------------------------------

SYNTHETIC_POLICY_PAGES: list[str] = [
    # Page 1
    "SYNTHETIC TEST POLICY DOCUMENT\n"
    "This document is fabricated content used only for automated testing "
    "of the retrieval pipeline. It does not represent a real insurance "
    "product. Definitions: 'Insured' means the person named in the policy "
    "schedule. 'Policy Year' means a period of twelve months from the "
    "policy inception date.",
    # Page 2
    "ROOM RENT LIMIT\n"
    "The Company shall pay for room, boarding and nursing expenses as "
    "provided by the hospital up to a maximum of one percent of the Sum "
    "Insured per day, subject to a cap of INR 5000 per day for a standard "
    "private room. Charges above this limit are the insured person's own "
    "responsibility.",
    # Page 3
    "INTENSIVE CARE UNIT CHARGES\n"
    "Expenses incurred for Intensive Care Unit or Intensive Cardiac Care "
    "Unit admission are payable up to two percent of the Sum Insured per "
    "day, with no fixed rupee cap, provided the treating doctor certifies "
    "the medical necessity of ICU admission.",
    # Page 4
    "CO-PAYMENT FOR SENIOR CITIZENS\n"
    "A mandatory co-payment of ten percent applies to all admissible "
    "claims for insured persons aged above sixty years at the time of "
    "hospitalization. This co-payment is calculated on the total "
    "admissible claim amount before any other deductions.",
    # Page 5
    "PERMANENT EXCLUSIONS\n"
    "The Company shall not be liable for any claim arising from cosmetic "
    "or aesthetic treatment, dental treatment unless arising from an "
    "accident, and treatment for obesity or weight control.",
    # Page 6
    "CLAIMS PROCEDURE\n"
    "The insured must notify the Company within twenty-four hours of "
    "emergency hospitalization. All original bills, the discharge "
    "summary, and investigation reports must be submitted within fifteen "
    "days of discharge for reimbursement claims to be processed.",
]


@pytest.fixture(scope="session")
def embedding_model():
    """Loads the local embedding model once per test session (slow to load)."""
    from rag.embeddings import EmbeddingModel

    return EmbeddingModel()


@pytest.fixture
def synthetic_policy_pdf(tmp_path: Path) -> Path:
    """A multi-page SYNTHETIC policy PDF with one clause per page."""
    pdf_path = tmp_path / "synthetic_policy.pdf"
    return _make_pdf(pdf_path, SYNTHETIC_POLICY_PAGES)


@pytest.fixture
def policy_vector_store(embedding_model, synthetic_policy_pdf: Path):
    """A ChunkVectorStore pre-populated with the synthetic policy's chunks."""
    from ingestion.chunker import chunk_pages
    from ingestion.pdf_parser import parse_pdf
    from rag.vector_store import ChunkVectorStore

    pages = parse_pdf(synthetic_policy_pdf, document_id="synthetic-policy-001")
    chunks = chunk_pages(pages, chunk_size=1200, chunk_overlap=200)

    store = ChunkVectorStore(
        collection_name=f"test_{synthetic_policy_pdf.stem}",
        embedding_model=embedding_model,
    )
    store.add_chunks(chunks)
    return store


@pytest.fixture
def empty_vector_store(embedding_model):
    """A ChunkVectorStore with nothing indexed."""
    from rag.vector_store import ChunkVectorStore

    return ChunkVectorStore(collection_name="test_empty_store", embedding_model=embedding_model)
