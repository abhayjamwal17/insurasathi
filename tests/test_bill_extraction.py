"""Tests for the Phase 4 hospital bill line-item extraction layer.

Uses StaticProvider/SequentialProvider (test doubles implementing
extraction.providers.LLMProvider) to exercise the pipeline deterministically,
without a network call or a running LLM. What's under test: category
control ("unknown" instead of a guess), missing-amount handling, page
provenance across a multi-page bill, and graceful handling of a poorly
structured page.
"""

from __future__ import annotations

import json

import pytest

from extraction.bill_extractor import BillExtractionError, BillLineItemExtractor
from extraction.bill_schemas import BILL_LINE_ITEM_CATEGORIES
from extraction.providers import LLMProvider, StaticProvider
from ingestion.schemas import DocumentPage


class SequentialProvider(LLMProvider):
    """Returns each queued response in order, one per call. Test double only."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(user_prompt)
        if not self._responses:
            raise AssertionError("SequentialProvider ran out of queued responses")
        return self._responses.pop(0)


def _page(page_number: int, text: str, filename: str = "bill.pdf") -> DocumentPage:
    return DocumentPage(document_id="bill-doc-001", filename=filename, page_number=page_number, text=text)


def _bill_response(items: list[dict]) -> str:
    return json.dumps({"line_items": items})


def test_clean_bill_extracts_a_single_confident_line_item() -> None:
    page = _page(1, "Room Rent (Private, 3 days) ......... INR 15000")
    response = _bill_response(
        [
            {
                "description": "Room Rent (Private, 3 days)",
                "amount": 15000,
                "quantity": 3,
                "unit_price": 5000,
                "category": "room_rent",
                "confidence": 0.95,
            }
        ]
    )
    extractor = BillLineItemExtractor(StaticProvider(response))

    items = extractor.extract_page(page)

    assert len(items) == 1
    item = items[0]
    assert item.description == "Room Rent (Private, 3 days)"
    assert item.amount == 15000
    assert item.quantity == 3
    assert item.unit_price == 5000
    assert item.category == "room_rent"
    assert item.page_number == 1


def test_multiple_line_items_on_one_page_are_all_extracted() -> None:
    page = _page(
        2,
        "Nursing Charges - INR 2000\nConsultant Visit Fee - INR 3000\nOT Charges - INR 40000",
    )
    response = _bill_response(
        [
            {"description": "Nursing Charges", "amount": 2000, "quantity": None,
             "unit_price": None, "category": "nursing", "confidence": 0.9},
            {"description": "Consultant Visit Fee", "amount": 3000, "quantity": None,
             "unit_price": None, "category": "consultant_doctor_charges", "confidence": 0.9},
            {"description": "OT Charges", "amount": 40000, "quantity": None,
             "unit_price": None, "category": "ot_charges", "confidence": 0.85},
        ]
    )
    extractor = BillLineItemExtractor(StaticProvider(response))

    items = extractor.extract_page(page)

    assert len(items) == 3
    assert {i.category for i in items} == {"nursing", "consultant_doctor_charges", "ot_charges"}
    assert all(i.page_number == 2 for i in items)


def test_unrecognizable_item_is_categorized_unknown_not_guessed() -> None:
    page = _page(3, "Miscellaneous Hospital Charges - INR 750")
    response = _bill_response(
        [
            {
                "description": "Miscellaneous Hospital Charges",
                "amount": 750,
                "quantity": None,
                "unit_price": None,
                "category": "unknown",
                "confidence": 0.3,
            }
        ]
    )
    extractor = BillLineItemExtractor(StaticProvider(response))

    items = extractor.extract_page(page)

    assert items[0].category == "unknown"


def test_hallucinated_category_is_forced_to_unknown() -> None:
    page = _page(3, "Some ambiguous line item")
    response = _bill_response(
        [
            {
                "description": "Some ambiguous line item",
                "amount": 100,
                "quantity": None,
                "unit_price": None,
                "category": "not_a_real_category",
                "confidence": 0.4,
            }
        ]
    )
    extractor = BillLineItemExtractor(StaticProvider(response))

    items = extractor.extract_page(page)

    assert items[0].category == "unknown"
    assert items[0].category in BILL_LINE_ITEM_CATEGORIES


def test_missing_amount_is_null_not_invented() -> None:
    page = _page(4, "Administrative Charges - see attached schedule")
    response = _bill_response(
        [
            {
                "description": "Administrative Charges",
                "amount": None,
                "quantity": None,
                "unit_price": None,
                "category": "administrative_charges",
                "confidence": 0.6,
            }
        ]
    )
    extractor = BillLineItemExtractor(StaticProvider(response))

    items = extractor.extract_page(page)

    assert items[0].amount is None
    assert items[0].description == "Administrative Charges"


def test_poorly_structured_page_with_no_items_returns_empty_list() -> None:
    page = _page(5, "***  HOSPITAL LETTERHEAD  ***\n(this page has no itemized charges)")
    response = _bill_response([])
    extractor = BillLineItemExtractor(StaticProvider(response))

    items = extractor.extract_page(page)

    assert items == []


def test_blank_page_short_circuits_without_calling_the_llm() -> None:
    page = _page(6, "   ")
    provider = SequentialProvider([])  # would raise if called
    extractor = BillLineItemExtractor(provider)

    items = extractor.extract_page(page)

    assert items == []
    assert provider.calls == []


def test_page_provenance_is_correct_across_a_multi_page_bill() -> None:
    page1 = _page(1, "Room Rent - INR 5000", filename="interim_bill.pdf")
    page2 = _page(2, "Medicines - INR 1200", filename="interim_bill.pdf")

    responses = [
        _bill_response([{"description": "Room Rent", "amount": 5000, "quantity": None,
                          "unit_price": None, "category": "room_rent", "confidence": 0.9}]),
        _bill_response([{"description": "Medicines", "amount": 1200, "quantity": None,
                          "unit_price": None, "category": "medicines", "confidence": 0.9}]),
    ]
    extractor = BillLineItemExtractor(SequentialProvider(responses))

    bill = extractor.extract_bill([page1, page2], document_id="bill-doc-001", filename="interim_bill.pdf")

    assert len(bill.line_items) == 2
    room_rent_item = next(i for i in bill.line_items if i.category == "room_rent")
    medicine_item = next(i for i in bill.line_items if i.category == "medicines")
    assert room_rent_item.page_number == 1
    assert medicine_item.page_number == 2
    assert room_rent_item.source_text == "Room Rent - INR 5000"
    assert medicine_item.source_text == "Medicines - INR 1200"


def test_malformed_json_raises_bill_extraction_error() -> None:
    page = _page(1, "Some charge - INR 100")
    extractor = BillLineItemExtractor(StaticProvider("not json at all"))

    with pytest.raises(BillExtractionError):
        extractor.extract_page(page)


def test_missing_line_items_key_raises_bill_extraction_error() -> None:
    page = _page(1, "Some charge - INR 100")
    extractor = BillLineItemExtractor(StaticProvider(json.dumps({"items": []})))

    with pytest.raises(BillExtractionError):
        extractor.extract_page(page)


def test_line_item_without_description_raises_bill_extraction_error() -> None:
    page = _page(1, "Some charge - INR 100")
    response = _bill_response([{"description": "", "amount": 100, "category": "unknown"}])
    extractor = BillLineItemExtractor(StaticProvider(response))

    with pytest.raises(BillExtractionError):
        extractor.extract_page(page)
