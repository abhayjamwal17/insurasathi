"""Tests for the Phase 6 evidence / confidence / human-review layer."""

from __future__ import annotations

from calculation.schemas import RoomRentCalculationStatus
from evidence.audit import audit_room_rent_charge
from evidence.schemas import ConfidenceLevel, ReviewStatus
from extraction.bill_schemas import BillLineItem, ExtractedBill
from extraction.schemas import ExtractedPolicyFields, FieldStatus, PolicyField, SourceEvidence


def _field(status: FieldStatus, value=None, confidence=None, page=None, notes=None) -> PolicyField:
    sources = []
    if page is not None:
        sources.append(
            SourceEvidence(
                document_id="doc1", filename="policy.pdf", page_number=page, chunk_id="c1", text="clause text"
            )
        )
    return PolicyField(status=status, value=value, confidence=confidence, sources=sources, notes=notes)


def _bill(items: list[BillLineItem]) -> ExtractedBill:
    return ExtractedBill(document_id="bill1", filename="bill.pdf", line_items=items)


def _room_rent_item(amount: float | None, page: int = 5) -> BillLineItem:
    return BillLineItem(
        description="Room Rent",
        amount=amount,
        category="room_rent",
        page_number=page,
        source_text="Room Rent - INR " + str(amount),
        confidence=0.9,
    )


def _complete_fields(**overrides) -> ExtractedPolicyFields:
    base = dict(
        room_rent_cap=_field(FieldStatus.OK, 5000, confidence=0.9, page=2),
        room_rent_cap_unit=_field(FieldStatus.OK, "INR_per_day", confidence=0.9, page=2),
        proportionate_deduction_applicable=_field(FieldStatus.OK, True, confidence=0.85, page=3),
        copay_percentage=PolicyField.unknown(),
        deductible=PolicyField.unknown(),
        relevant_sub_limits=PolicyField.unknown(),
        potentially_relevant_exclusions=PolicyField.unknown(),
    )
    base.update(overrides)
    return ExtractedPolicyFields(**base)


def test_valid_calculation_returns_ok_and_green() -> None:
    fields = _complete_fields()
    bill = _bill([_room_rent_item(10000)])

    result = audit_room_rent_charge(fields, bill, actual_room_rent_per_day=10000)

    assert result.status == ReviewStatus.OK
    assert result.confidence_level == ConfidenceLevel.GREEN
    assert result.estimated_result is not None
    assert result.estimated_result.status == RoomRentCalculationStatus.OK
    assert result.applied_rules == ["room_rent_cap_proportionate_deduction"]
    assert len(result.evidence) == 3  # cap, unit, proportionate flag
    assert result.reason is None


def test_missing_policy_evidence_cannot_determine() -> None:
    fields = _complete_fields(room_rent_cap=PolicyField.unknown(), room_rent_cap_unit=PolicyField.unknown())
    bill = _bill([_room_rent_item(10000)])

    result = audit_room_rent_charge(fields, bill, actual_room_rent_per_day=10000)

    assert result.status == ReviewStatus.CANNOT_DETERMINE
    assert result.confidence_level == ConfidenceLevel.RED
    assert result.estimated_result is None
    assert "missing" in result.reason.lower()


def test_conflicting_clauses_needs_review() -> None:
    fields = _complete_fields(
        room_rent_cap=_field(
            FieldStatus.NEEDS_REVIEW,
            value=None,
            page=2,
            notes="Base policy states INR 5000/day; endorsement states INR 7500/day.",
        )
    )
    bill = _bill([_room_rent_item(10000)])

    result = audit_room_rent_charge(fields, bill, actual_room_rent_per_day=10000)

    assert result.status == ReviewStatus.NEEDS_REVIEW
    assert result.confidence_level == ConfidenceLevel.AMBER
    assert result.estimated_result is None  # never calculated on conflicting evidence
    assert "conflict" in result.reason.lower()
    assert len(result.evidence) == 1


def test_low_confidence_extraction_needs_review() -> None:
    fields = _complete_fields(
        room_rent_cap=_field(FieldStatus.OK, 5000, confidence=0.2, page=2),
    )
    bill = _bill([_room_rent_item(10000)])

    result = audit_room_rent_charge(fields, bill, actual_room_rent_per_day=10000)

    assert result.status == ReviewStatus.NEEDS_REVIEW
    assert result.confidence_level == ConfidenceLevel.AMBER
    # Calculation still ran (it's just flagged), so the estimate is still shown for review.
    assert result.estimated_result is not None
    assert any("confidence" in w.lower() for w in result.warnings)


def test_unknown_bill_item_generates_warning_without_blocking() -> None:
    fields = _complete_fields()
    unknown_item = BillLineItem(
        description="Miscellaneous Charges",
        amount=500,
        category="unknown",
        page_number=6,
        source_text="Miscellaneous Charges - INR 500",
    )
    bill = _bill([_room_rent_item(10000), unknown_item])

    result = audit_room_rent_charge(fields, bill, actual_room_rent_per_day=10000)

    assert result.status == ReviewStatus.OK
    assert any("unknown" in w.lower() for w in result.warnings)


def test_missing_bill_room_rent_item_cannot_determine() -> None:
    fields = _complete_fields()
    bill = _bill(
        [
            BillLineItem(
                description="Medicines", amount=1000, category="medicines",
                page_number=4, source_text="Medicines - INR 1000",
            )
        ]
    )

    result = audit_room_rent_charge(fields, bill, actual_room_rent_per_day=10000)

    assert result.status == ReviewStatus.CANNOT_DETERMINE
    assert "bill extraction is insufficient" in result.reason.lower()


def test_successful_human_escalation_state_when_associated_charge_scope_unresolved() -> None:
    fields = _complete_fields(
        proportionate_deduction_applicable=PolicyField.unknown(),
    )
    bill = _bill([_room_rent_item(10000)])

    result = audit_room_rent_charge(fields, bill, actual_room_rent_per_day=10000)

    assert result.status == ReviewStatus.NEEDS_REVIEW
    assert result.confidence_level == ConfidenceLevel.AMBER
    assert result.estimated_result is not None
    assert result.estimated_result.status == RoomRentCalculationStatus.NEEDS_REVIEW
    # The room-rent-itself adjustment is still shown, so a reviewer has something to act on.
    assert result.estimated_result.room_rent_adjustment is not None
    assert result.reason is not None
    assert len(result.evidence) > 0


def test_unsupported_unit_is_cannot_determine() -> None:
    fields = _complete_fields(
        room_rent_cap_unit=_field(FieldStatus.OK, "percent_of_sum_insured_per_day", confidence=0.9, page=2),
    )
    bill = _bill([_room_rent_item(10000)])

    result = audit_room_rent_charge(fields, bill, actual_room_rent_per_day=10000)

    assert result.status == ReviewStatus.CANNOT_DETERMINE
    assert "applicable rule cannot be established" in result.reason.lower()


def test_missing_actual_room_rent_is_cannot_determine() -> None:
    fields = _complete_fields()
    bill = _bill([_room_rent_item(10000)])

    result = audit_room_rent_charge(fields, bill, actual_room_rent_per_day=None)

    assert result.status == ReviewStatus.CANNOT_DETERMINE
