"""Turns retrieved evidence into structured, provenance-carrying policy fields.

Flow: Evidence -> LLM interpretation -> structured field. The extractor
never lets the LLM assert its own page/document metadata — it only trusts
the LLM to say *which chunk_id(s)* it used, then looks those chunks up in
the evidence that was actually supplied and builds SourceEvidence from
that. This is what keeps page citations trustworthy even though the LLM
output itself is not.
"""

from __future__ import annotations

import json

from extraction.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt
from extraction.providers import LLMProvider, LLMProviderError
from extraction.schemas import (
    FIELD_NAMES,
    ExtractedPolicyFields,
    FieldStatus,
    PolicyField,
    SourceEvidence,
)
from rag.schemas import RetrievedChunk


class ExtractionError(Exception):
    """Raised when the LLM's output cannot be turned into valid fields."""


def _source_evidence_for(chunk: RetrievedChunk) -> SourceEvidence:
    return SourceEvidence(
        document_id=chunk.document_id,
        filename=chunk.filename,
        page_number=chunk.page_number,
        chunk_id=chunk.chunk_id,
        text=chunk.text,
    )


def _parse_field(name: str, raw: dict, chunks_by_id: dict[str, RetrievedChunk]) -> PolicyField:
    if not isinstance(raw, dict):
        raise ExtractionError(f"Field '{name}': expected an object, got {type(raw).__name__}")

    try:
        status = FieldStatus(raw.get("status"))
    except ValueError as exc:
        raise ExtractionError(f"Field '{name}': invalid status {raw.get('status')!r}") from exc

    requested_ids = raw.get("source_chunk_ids") or []
    if not isinstance(requested_ids, list):
        raise ExtractionError(f"Field '{name}': source_chunk_ids must be a list")

    # Only trust chunk_ids that were genuinely part of the supplied evidence.
    sources = [_source_evidence_for(chunks_by_id[cid]) for cid in requested_ids if cid in chunks_by_id]

    value = raw.get("value")
    confidence = raw.get("confidence")
    notes = raw.get("notes")

    if status in (FieldStatus.UNKNOWN, FieldStatus.NEEDS_REVIEW):
        # Never let a value survive alongside unknown/needs_review — those
        # statuses exist specifically so no value is asserted.
        value = None
        confidence = None

    return PolicyField(
        status=status,
        value=value,
        confidence=confidence,
        sources=sources,
        notes=notes,
    )


class PolicyExtractor:
    """Runs the evidence -> LLM -> structured-field pipeline for one query."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def extract(self, chunks: list[RetrievedChunk]) -> ExtractedPolicyFields:
        """Extract all policy fields from a set of retrieved evidence chunks.

        If `chunks` is empty, every field is returned as unknown without
        calling the LLM at all — there is nothing for it to interpret.
        """
        if not chunks:
            return ExtractedPolicyFields.all_unknown(notes="No evidence was retrieved.")

        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

        user_prompt = build_extraction_prompt(chunks)
        raw_text = self._provider.complete_json(EXTRACTION_SYSTEM_PROMPT, user_prompt)

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"LLM output was not valid JSON: {exc}\nRaw output: {raw_text!r}") from exc

        if not isinstance(data, dict):
            raise ExtractionError(f"LLM output was not a JSON object: {raw_text!r}")

        missing = [name for name in FIELD_NAMES if name not in data]
        if missing:
            raise ExtractionError(f"LLM output is missing fields: {missing}")

        fields = {name: _parse_field(name, data[name], chunks_by_id) for name in FIELD_NAMES}
        return ExtractedPolicyFields(**fields)
