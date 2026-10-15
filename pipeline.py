"""End-to-end orchestration: Policy + Bill -> RAG -> LLM -> JSON -> Calculation -> Evidence -> Result.

Pure Python, no Streamlit dependency, so the exact same functions drive
both `streamlit_app.py` and `pipeline_demo.py` (the offline phase-gate
demonstration). Every stage returns a StageResult instead of raising, so
one stage failing (a bad PDF, a down LLM provider, insufficient retrieval
evidence, ...) never crashes the caller — it's surfaced as a clear message
and later stages are simply skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import settings
from evidence.audit import audit_room_rent_charge
from evidence.schemas import AuditResult
from extraction.bill_extractor import BillExtractionError, BillLineItemExtractor
from extraction.bill_schemas import ExtractedBill
from extraction.extractor import ExtractionError, PolicyExtractor
from extraction.providers import LLMProvider, LLMProviderError
from extraction.schemas import ExtractedPolicyFields
from ingestion.chunker import chunk_pages
from ingestion.pdf_parser import PDFParseError, parse_pdf
from ingestion.schemas import DocumentPage
from rag.retriever import Retriever
from rag.schemas import RetrievedChunk
from rag.vector_store import ChunkVectorStore

# Evidence for these questions is gathered and handed to the LLM extractor
# together, since ExtractedPolicyFields covers all of them at once.
POLICY_EVIDENCE_QUERIES: tuple[str, ...] = (
    "room rent limit per day",
    "proportionate deduction of other expenses when room rent exceeds the limit",
    "co-payment percentage",
    "deductible amount",
    "sub-limits such as ICU charges",
    "exclusions",
)


@dataclass
class StageResult:
    """Generic per-stage outcome: never raises past the boundary that calls it."""

    ok: bool
    message: str = ""
    data: Any = None


def ingest_pdf(path: str | Path, document_id: str) -> StageResult:
    """Stage: page-aware PDF extraction. Handles invalid/missing/empty PDFs."""
    try:
        pages = parse_pdf(path, document_id=document_id)
        return StageResult(True, f"Parsed {len(pages)} page(s).", pages)
    except PDFParseError as exc:
        return StageResult(False, f"Invalid PDF: {exc}")


def retrieve_policy_evidence(
    policy_pages: list[DocumentPage],
    top_k: int | None = None,
    relevance_threshold: float | None = None,
) -> StageResult:
    """Stage: chunk + index the policy, then run retrieval for each evidence question."""
    top_k = top_k if top_k is not None else settings.retrieval_top_k
    relevance_threshold = (
        relevance_threshold if relevance_threshold is not None else settings.retrieval_relevance_threshold
    )
    try:
        chunks = chunk_pages(policy_pages, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        # A fresh, in-memory collection per run keeps uploads isolated from each other.
        store = ChunkVectorStore(collection_name=f"session_{id(policy_pages)}")
        store.add_chunks(chunks)
        retriever = Retriever(store, top_k=top_k, relevance_threshold=relevance_threshold)

        per_query: dict[str, Any] = {}
        combined_chunks: dict[str, RetrievedChunk] = {}
        for query in POLICY_EVIDENCE_QUERIES:
            result = retriever.retrieve(query)
            per_query[query] = result
            for chunk in result.chunks:
                combined_chunks[chunk.chunk_id] = chunk

        evidence_chunks = list(combined_chunks.values())
        if not evidence_chunks:
            return StageResult(
                False,
                "Insufficient evidence: no policy clause cleared the relevance threshold for any "
                "of the evidence questions.",
                {"per_query": per_query, "chunks": []},
            )

        return StageResult(
            True,
            f"Retrieved {len(evidence_chunks)} unique evidence chunk(s) across "
            f"{len(POLICY_EVIDENCE_QUERIES)} questions.",
            {"per_query": per_query, "chunks": evidence_chunks},
        )
    except Exception as exc:  # retrieval must never take the whole app down
        return StageResult(False, f"Retrieval failed: {exc}")


def extract_policy_fields(provider: LLMProvider, evidence_chunks: list[RetrievedChunk]) -> StageResult:
    """Stage: LLM interpretation of retrieved evidence into ExtractedPolicyFields."""
    try:
        fields: ExtractedPolicyFields = PolicyExtractor(provider).extract(evidence_chunks)
        return StageResult(True, "Policy field extraction complete.", fields)
    except ExtractionError as exc:
        return StageResult(False, f"Policy extraction failed: {exc}")
    except LLMProviderError as exc:
        return StageResult(False, f"LLM provider error during policy extraction: {exc}")


def extract_bill_line_items(
    provider: LLMProvider, bill_pages: list[DocumentPage], document_id: str, filename: str
) -> StageResult:
    """Stage: LLM interpretation of bill pages into categorized line items."""
    try:
        bill: ExtractedBill = BillLineItemExtractor(provider).extract_bill(
            bill_pages, document_id=document_id, filename=filename
        )
        return StageResult(True, f"Extracted {len(bill.line_items)} line item(s).", bill)
    except BillExtractionError as exc:
        return StageResult(False, f"Bill extraction failed: {exc}")
    except LLMProviderError as exc:
        return StageResult(False, f"LLM provider error during bill extraction: {exc}")


def run_room_rent_audit(
    policy_fields: ExtractedPolicyFields,
    bill: ExtractedBill,
    actual_room_rent_per_day: float | None,
) -> StageResult:
    """Stage: deterministic calculation + evidence/confidence/abstention (no LLM)."""
    try:
        result: AuditResult = audit_room_rent_charge(policy_fields, bill, actual_room_rent_per_day)
        return StageResult(True, f"Audit status: {result.status.value}", result)
    except Exception as exc:  # calculation must never crash the app either
        return StageResult(False, f"Calculation failed: {exc}")
