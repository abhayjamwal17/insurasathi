"""Minimal Streamlit UI exposing the technical pipeline end to end.

This is NOT the product UI — the polished experience is the separate
Claude Design prototype. This app exists only to demonstrate, stage by
stage, that the underlying architecture (RAG -> LLM -> structured JSON ->
deterministic calculation -> evidence/confidence) actually works.

Run with: streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

import pipeline
from config import settings
from evidence.schemas import ConfidenceLevel, ReviewStatus
from extraction.providers import LLMProviderError, get_provider

st.set_page_config(page_title="ICU / Hospital Interim Bill Auditor", layout="wide")

st.title("ICU / HOSPITAL INTERIM BILL AUDITOR")
st.error("**DEMO / PROTOTYPE — NOT AN INSURANCE CLAIM DECISION**")
st.caption(
    "Technical proof-of-concept only. The polished product UI is a separate Claude Design "
    "prototype — this page exists to show that the underlying pipeline actually works."
)

with st.expander("How this maps to the architecture", expanded=False):
    st.markdown(
        "```\n"
        "Policy PDF + Bill PDF\n"
        "        |\n"
        "Page-aware extraction + chunking\n"
        "        |\n"
        "RAG retrieval  (vector search)\n"
        "        |\n"
        "LLM interpretation  <-- AI / RAG stops here\n"
        "        |\n"
        "STRUCTURED JSON  <-- the boundary\n"
        "        |\n"
        "Deterministic calculation  <-- plain Python, no LLM, from here down\n"
        "        |\n"
        "Evidence + confidence + human review\n"
        "```"
    )

col1, col2 = st.columns(2)
with col1:
    policy_file = st.file_uploader("Insurance Policy PDF", type=["pdf"], key="policy_upload")
with col2:
    bill_file = st.file_uploader("Hospital Interim Bill PDF", type=["pdf"], key="bill_upload")

actual_room_rent_per_day = st.number_input(
    "Actual room rent charged (INR per day) — required for the room-rent calculation. "
    "The system will not infer this from the bill total; enter it explicitly.",
    min_value=0.0,
    value=0.0,
    step=100.0,
)

analyze_clicked = st.button("Analyze", type="primary")


def _confidence_badge(level: ConfidenceLevel) -> None:
    if level == ConfidenceLevel.GREEN:
        st.success(f"Confidence: {level.value} — Verified / High confidence")
    elif level == ConfidenceLevel.AMBER:
        st.warning(f"Confidence: {level.value} — Needs review")
    else:
        st.error(f"Confidence: {level.value} — Cannot determine / escalation required")


def _status_badge(status: ReviewStatus) -> None:
    if status == ReviewStatus.OK:
        st.success(f"Review status: {status.value}")
    elif status == ReviewStatus.NEEDS_REVIEW:
        st.warning(f"Review status: {status.value}")
    else:
        st.error(f"Review status: {status.value}")


if not analyze_clicked:
    st.info("Upload a policy PDF and a hospital bill PDF, then click Analyze.")
    st.stop()

# --- missing-file error handling ---
if not policy_file:
    st.error("Please upload an Insurance Policy PDF.")
    st.stop()
if not bill_file:
    st.error("Please upload a Hospital Interim Bill PDF.")
    st.stop()

with tempfile.TemporaryDirectory() as tmp_dir:
    policy_path = Path(tmp_dir) / policy_file.name
    policy_path.write_bytes(policy_file.getvalue())
    bill_path = Path(tmp_dir) / bill_file.name
    bill_path.write_bytes(bill_file.getvalue())

    # ===================================================================
    # 1. Document status
    # ===================================================================
    st.header("1. Document Status")
    doc_col1, doc_col2 = st.columns(2)

    policy_pages_result = pipeline.ingest_pdf(policy_path, document_id="policy-upload")
    with doc_col1:
        if policy_pages_result.ok:
            st.success(f"Policy PDF: {policy_pages_result.message}")
        else:
            st.error(f"Policy PDF: {policy_pages_result.message}")

    bill_pages_result = pipeline.ingest_pdf(bill_path, document_id="bill-upload")
    with doc_col2:
        if bill_pages_result.ok:
            st.success(f"Bill PDF: {bill_pages_result.message}")
        else:
            st.error(f"Bill PDF: {bill_pages_result.message}")

    if not policy_pages_result.ok or not bill_pages_result.ok:
        st.warning("Cannot continue: fix the invalid PDF(s) above and re-run.")
        st.stop()

    st.markdown("### AI / RAG interpretation")
    st.caption("Everything in this section is produced by retrieval + an LLM. Nothing here is a financial calculation.")

    # ===================================================================
    # 2. Retrieved policy evidence
    # ===================================================================
    st.header("2. Retrieved Policy Evidence")
    retrieval_result = pipeline.retrieve_policy_evidence(policy_pages_result.data)
    if retrieval_result.ok:
        st.success(retrieval_result.message)
        for query, result in retrieval_result.data["per_query"].items():
            with st.expander(f"Query: {query!r} — {result.status}"):
                if result.status == "insufficient_evidence":
                    st.warning("Insufficient evidence for this question.")
                for chunk in result.chunks:
                    st.markdown(
                        f"**page {chunk.page_number}** (score={chunk.relevance_score:.3f}, "
                        f"chunk_id=`{chunk.chunk_id}`)"
                    )
                    st.text(chunk.text)
    else:
        st.warning(retrieval_result.message)

    evidence_chunks = retrieval_result.data["chunks"] if retrieval_result.data else []

    # ===================================================================
    # 3. Extracted policy variables
    # ===================================================================
    st.header("3. Extracted Policy Variables")
    try:
        provider = get_provider(
            settings.llm_provider,
            groq_api_key=settings.groq_api_key,
            groq_model=settings.groq_model,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
        )
        provider_error: str | None = None
    except LLMProviderError as exc:
        provider = None
        provider_error = str(exc)

    policy_extraction_result = None
    if provider is None:
        st.error(
            f"LLM provider unavailable ({settings.llm_provider}): {provider_error}. "
            "Set LLM_PROVIDER/GROQ_API_KEY or run a local Ollama server in .env."
        )
    else:
        policy_extraction_result = pipeline.extract_policy_fields(provider, evidence_chunks)
        if policy_extraction_result.ok:
            fields = policy_extraction_result.data
            rows = []
            for name in fields.model_fields:
                field = getattr(fields, name)
                rows.append(
                    {
                        "field": name,
                        "status": field.status.value,
                        "value": field.value,
                        "confidence": field.confidence,
                        "pages": ", ".join(str(s.page_number) for s in field.sources) or "-",
                    }
                )
            st.table(rows)
        else:
            st.error(policy_extraction_result.message)

    # ===================================================================
    # 4. Extracted bill line items
    # ===================================================================
    st.header("4. Extracted Bill Line Items")
    bill_extraction_result = None
    if provider is None:
        st.error("Skipped: no LLM provider available (see section 3).")
    else:
        bill_extraction_result = pipeline.extract_bill_line_items(
            provider, bill_pages_result.data, document_id="bill-upload", filename=bill_file.name
        )
        if bill_extraction_result.ok:
            bill = bill_extraction_result.data
            if bill.line_items:
                st.table(
                    [
                        {
                            "description": item.description,
                            "amount": item.amount,
                            "category": item.category,
                            "page": item.page_number,
                            "confidence": item.confidence,
                        }
                        for item in bill.line_items
                    ]
                )
            else:
                st.warning("No line items were extracted from the bill.")
        else:
            st.error(bill_extraction_result.message)

    # ===================================================================
    # 5. AI -> Calculation Engine JSON  (the architectural boundary)
    # ===================================================================
    st.header("5. AI → Calculation Engine JSON")
    st.info(
        "This is the exact boundary the architecture requires: everything above this line was "
        "produced by AI/RAG. Everything below is plain, deterministic Python arithmetic over "
        "this JSON — the LLM is never called again after this point."
    )
    if policy_extraction_result and policy_extraction_result.ok:
        st.markdown("**Extracted policy fields (JSON handed to the calculation engine):**")
        st.code(policy_extraction_result.data.model_dump_json(indent=2), language="json")
    if bill_extraction_result and bill_extraction_result.ok:
        st.markdown("**Extracted bill (JSON handed to the calculation engine):**")
        st.code(bill_extraction_result.data.model_dump_json(indent=2), language="json")

    if not (policy_extraction_result and policy_extraction_result.ok):
        st.warning("Cannot continue to calculation: policy extraction did not succeed.")
        st.stop()
    if not (bill_extraction_result and bill_extraction_result.ok):
        st.warning("Cannot continue to calculation: bill extraction did not succeed.")
        st.stop()

    st.markdown("---")
    st.markdown("### Deterministic calculation engine (no LLM calls below this point)")

    # ===================================================================
    # 6 & 7. Deterministic calculation + final estimate
    # ===================================================================
    st.header("6. Deterministic Calculation")
    audit_stage = pipeline.run_room_rent_audit(
        policy_extraction_result.data,
        bill_extraction_result.data,
        actual_room_rent_per_day if actual_room_rent_per_day > 0 else None,
    )

    if not audit_stage.ok:
        st.error(audit_stage.message)
        st.stop()

    audit_result = audit_stage.data

    if audit_result.estimated_result is not None:
        calc = audit_result.estimated_result
        st.write(f"Calculation status: `{calc.status.value}`")
        for step in calc.trace.steps:
            st.markdown(f"- **{step.label}:** {step.detail}")
    else:
        st.info("No calculation was performed — see status/reason below.")

    st.header("7. Final Estimate")
    if audit_result.estimated_result is not None and audit_result.estimated_result.total_adjustment is not None:
        calc = audit_result.estimated_result
        est_col1, est_col2, est_col3 = st.columns(3)
        est_col1.metric("Room rent adjustment", f"INR {calc.room_rent_adjustment:,.2f}")
        est_col2.metric("Associated charge adjustment", f"INR {calc.associated_charge_adjustment:,.2f}")
        est_col3.metric("Total adjustment", f"INR {calc.total_adjustment:,.2f}")
    else:
        st.info("No final estimate is available for this run.")

    # ===================================================================
    # 8. Evidence
    # ===================================================================
    st.header("8. Evidence")
    if audit_result.evidence:
        st.table(
            [
                {
                    "field": item.extracted_field,
                    "document": item.document,
                    "page": item.page,
                    "confidence": item.confidence,
                    "rule": item.rule_applied,
                    "source_text": item.source_text[:200],
                }
                for item in audit_result.evidence
            ]
        )
    else:
        st.info("No evidence was attached to this result.")

    # ===================================================================
    # 9. Confidence
    # ===================================================================
    st.header("9. Confidence")
    _confidence_badge(audit_result.confidence_level)

    # ===================================================================
    # 10. Warnings / human review
    # ===================================================================
    st.header("10. Warnings / Human Review")
    _status_badge(audit_result.status)
    if audit_result.reason:
        st.markdown(f"**Reason:** {audit_result.reason}")
    if audit_result.warnings:
        for warning in audit_result.warnings:
            st.warning(warning)
    else:
        st.caption("No additional warnings.")
