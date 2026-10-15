"""Turns page-aware hospital bill text into structured, categorized line items.

Pipeline: Hospital Bill PDF -> page-aware extraction (ingestion) ->
structured line-item extraction (this module, LLM-backed) -> Pydantic
validation -> normalized ExtractedBill.

This module only extracts and categorizes. It never decides whether an
item is covered by a policy, and it never computes totals — those are
later, deterministic phases.

Page provenance is authoritative from the source DocumentPage, not from
anything the LLM claims: every BillLineItem's page_number is set by this
module from the page it was asked to read, matching the provenance
discipline used in extraction/extractor.py for policy fields.
"""

from __future__ import annotations

import json

from extraction.bill_prompts import BILL_EXTRACTION_SYSTEM_PROMPT, build_bill_extraction_prompt
from extraction.bill_schemas import (
    BILL_LINE_ITEM_CATEGORIES,
    UNKNOWN_CATEGORY,
    BillLineItem,
    ExtractedBill,
)
from extraction.providers import LLMProvider
from ingestion.schemas import DocumentPage


class BillExtractionError(Exception):
    """Raised when the LLM's output cannot be turned into valid line items."""


def _coerce_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _parse_line_item(raw: dict, page: DocumentPage) -> BillLineItem:
    if not isinstance(raw, dict):
        raise BillExtractionError(f"Expected a line item object, got {type(raw).__name__}")

    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        raise BillExtractionError(f"Line item missing a non-empty description: {raw!r}")

    category = raw.get("category")
    if category not in BILL_LINE_ITEM_CATEGORIES:
        # Never trust a hallucinated/out-of-list category: fall back to
        # "unknown" rather than raising, matching "do not guess."
        category = UNKNOWN_CATEGORY

    confidence = raw.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence = None

    return BillLineItem(
        description=description.strip(),
        amount=_coerce_number(raw.get("amount")),
        quantity=_coerce_number(raw.get("quantity")),
        unit_price=_coerce_number(raw.get("unit_price")),
        category=category,
        page_number=page.page_number,
        source_text=page.text,
        confidence=confidence,
    )


class BillLineItemExtractor:
    """Runs the bill-page -> LLM -> structured-line-items pipeline."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def extract_page(self, page: DocumentPage) -> list[BillLineItem]:
        """Extract line items from a single bill page. Empty list if none."""
        if not page.text.strip():
            return []

        user_prompt = build_bill_extraction_prompt(page)
        raw_text = self._provider.complete_json(BILL_EXTRACTION_SYSTEM_PROMPT, user_prompt)

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise BillExtractionError(
                f"LLM output was not valid JSON: {exc}\nRaw output: {raw_text!r}"
            ) from exc

        if not isinstance(data, dict) or "line_items" not in data:
            raise BillExtractionError(f"LLM output missing 'line_items': {raw_text!r}")

        raw_items = data["line_items"]
        if not isinstance(raw_items, list):
            raise BillExtractionError(f"'line_items' must be a list: {raw_text!r}")

        return [_parse_line_item(item, page) for item in raw_items]

    def extract_bill(
        self,
        pages: list[DocumentPage],
        document_id: str,
        filename: str,
    ) -> ExtractedBill:
        """Extract and normalize line items across every page of a bill."""
        line_items: list[BillLineItem] = []
        for page in pages:
            line_items.extend(self.extract_page(page))

        return ExtractedBill(document_id=document_id, filename=filename, line_items=line_items)
