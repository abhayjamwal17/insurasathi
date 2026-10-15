"""Phase 2 demonstration: retrieval over a SYNTHETIC test policy.

Builds a small, clearly-labeled synthetic policy PDF in memory (not a real
insurance document), ingests it through the Phase 1 pipeline, indexes it
into a Chroma vector store, and runs a handful of natural-language queries
against it — printing the retrieved evidence, its page number, and its
source document for each.

This demonstrates retrieval only. No LLM, no answer generation, no
calculation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import fitz

from ingestion.chunker import chunk_pages
from ingestion.pdf_parser import parse_pdf
from rag.retriever import Retriever
from rag.vector_store import ChunkVectorStore

# Clearly-labeled synthetic content — NOT a real insurance policy.
SYNTHETIC_POLICY_PAGES = [
    "SYNTHETIC TEST POLICY DOCUMENT\n"
    "This document is fabricated content used only to demonstrate the "
    "retrieval pipeline. It does not represent a real insurance product. "
    "Definitions: 'Insured' means the person named in the policy schedule.",
    "ROOM RENT LIMIT\n"
    "The Company shall pay for room, boarding and nursing expenses as "
    "provided by the hospital up to a maximum of one percent of the Sum "
    "Insured per day, subject to a cap of INR 5000 per day for a standard "
    "private room. Charges above this limit are the insured person's own "
    "responsibility.",
    "INTENSIVE CARE UNIT CHARGES\n"
    "Expenses incurred for Intensive Care Unit or Intensive Cardiac Care "
    "Unit admission are payable up to two percent of the Sum Insured per "
    "day, with no fixed rupee cap, provided the treating doctor certifies "
    "the medical necessity of ICU admission.",
    "CO-PAYMENT FOR SENIOR CITIZENS\n"
    "A mandatory co-payment of ten percent applies to all admissible "
    "claims for insured persons aged above sixty years at the time of "
    "hospitalization.",
    "PERMANENT EXCLUSIONS\n"
    "The Company shall not be liable for any claim arising from cosmetic "
    "or aesthetic treatment, dental treatment unless arising from an "
    "accident, and treatment for obesity or weight control.",
    "CLAIMS PROCEDURE\n"
    "The insured must notify the Company within twenty-four hours of "
    "emergency hospitalization. All original bills, the discharge "
    "summary, and investigation reports must be submitted within fifteen "
    "days of discharge for reimbursement claims to be processed.",
]

DEMO_QUERIES = [
    "What is the room rent limit?",
    "What percentage co-payment applies to senior citizens?",
    "Are cosmetic treatments covered?",
    "What is the capital of France?",  # expected: insufficient evidence
]


def _build_synthetic_policy_pdf(path: Path) -> None:
    doc = fitz.open()
    for text in SYNTHETIC_POLICY_PAGES:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 500, 750), text)
    doc.save(path)
    doc.close()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = Path(tmp_dir) / "synthetic_policy.pdf"
        _build_synthetic_policy_pdf(pdf_path)

        print("=" * 70)
        print("Indexing SYNTHETIC test policy (not a real insurance document)")
        print("=" * 70)

        pages = parse_pdf(pdf_path, document_id="synthetic-policy-demo")
        chunks = chunk_pages(pages)
        print(f"Parsed {len(pages)} pages -> {len(chunks)} chunks\n")

        store = ChunkVectorStore(collection_name="rag_demo")
        store.add_chunks(chunks)

        retriever = Retriever(store, top_k=3, relevance_threshold=0.3)

        for query in DEMO_QUERIES:
            result = retriever.retrieve(query)
            print(f"Query: {query!r}")
            if result.status == "insufficient_evidence":
                print("  -> INSUFFICIENT EVIDENCE (no chunk cleared the relevance threshold)")
            else:
                for chunk in result.chunks:
                    preview = " ".join(chunk.text.split())[:120]
                    print(f"  -> [score={chunk.relevance_score:.3f}] page {chunk.page_number} "
                          f"({chunk.filename}, chunk_id={chunk.chunk_id})")
                    print(f"     {preview}...")
            print()


if __name__ == "__main__":
    main()
