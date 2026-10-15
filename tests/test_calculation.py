"""Tests for the Phase 5 deterministic calculation engine.

No LLM calls anywhere in this module or the code under test — every case
supplies plain numeric/boolean inputs and checks a deterministic output.
"""

from __future__ import annotations

from calculation.room_rent import (
    apply_room_rent_rule,
    calculate_room_rent_proportionate_deduction,
)
from calculation.schemas import RoomRentCalculationStatus
from extraction.schemas import ExtractedPolicyFields, FieldStatus, PolicyField, SourceEvidence


def test_normal_room_rent_cap_applies_proportionate_deduction() -> None:
    result = calculate_room_rent_proportionate_deduction(
        policy_room_rent_cap=5000,
        policy_room_rent_cap_unit="INR_per_day",
        proportionate_deduction_applicable=True,
        actual_room_rent_per_day=10000,
        billed_room_rent_amount=30000,  # 3 days at 10000/day
        associated_charges_amount=9000,  # e.g. nursing charges for the same 3 days
        source_page=2,
    )

    assert result.status == RoomRentCalculationStatus.OK
    assert result.ratio == 0.5
    # room rent adjustment = 30000 * (1 - 0.5) = 15000
    assert result.room_rent_adjustment == 15000.0
    # associated adjustment = 9000 * (1 - 0.5) = 4500
    assert result.associated_charge_adjustment == 4500.0
    assert result.total_adjustment == 19500.0
    assert result.final_room_rent_amount == 15000.0
    assert result.source_page == 2
    assert len(result.trace.steps) > 0


def test_no_room_rent_deduction_when_within_cap() -> None:
    result = calculate_room_rent_proportionate_deduction(
        policy_room_rent_cap=5000,
        policy_room_rent_cap_unit="INR_per_day",
        proportionate_deduction_applicable=True,
        actual_room_rent_per_day=4000,
        billed_room_rent_amount=12000,
        associated_charges_amount=3000,
    )

    assert result.status == RoomRentCalculationStatus.NOT_APPLICABLE
    assert result.ratio == 1.0
    assert result.room_rent_adjustment == 0.0
    assert result.associated_charge_adjustment == 0.0
    assert result.total_adjustment == 0.0
    assert result.final_room_rent_amount == 12000.0


def test_proportionate_deduction_not_applicable_leaves_associated_charges_untouched() -> None:
    result = calculate_room_rent_proportionate_deduction(
        policy_room_rent_cap=5000,
        policy_room_rent_cap_unit="INR_per_day",
        proportionate_deduction_applicable=False,
        actual_room_rent_per_day=10000,
        billed_room_rent_amount=10000,
        associated_charges_amount=8000,
    )

    assert result.status == RoomRentCalculationStatus.OK
    assert result.room_rent_adjustment == 5000.0  # 10000 * (1 - 0.5)
    assert result.associated_charge_adjustment == 0.0


def test_unknown_proportionate_deduction_flag_needs_review() -> None:
    result = calculate_room_rent_proportionate_deduction(
        policy_room_rent_cap=5000,
        policy_room_rent_cap_unit="INR_per_day",
        proportionate_deduction_applicable=None,
        actual_room_rent_per_day=10000,
        billed_room_rent_amount=10000,
        associated_charges_amount=8000,
    )

    assert result.status == RoomRentCalculationStatus.NEEDS_REVIEW
    assert result.room_rent_adjustment == 5000.0
    assert result.associated_charge_adjustment == 0.0


def test_zero_actual_room_rent_is_invalid() -> None:
    result = calculate_room_rent_proportionate_deduction(
        policy_room_rent_cap=5000,
        policy_room_rent_cap_unit="INR_per_day",
        proportionate_deduction_applicable=True,
        actual_room_rent_per_day=0,
        billed_room_rent_amount=0,
    )

    assert result.status == RoomRentCalculationStatus.INSUFFICIENT_INPUT
    assert result.ratio is None


def test_negative_policy_cap_is_invalid() -> None:
    result = calculate_room_rent_proportionate_deduction(
        policy_room_rent_cap=-100,
        policy_room_rent_cap_unit="INR_per_day",
        proportionate_deduction_applicable=True,
        actual_room_rent_per_day=6000,
        billed_room_rent_amount=6000,
    )

    assert result.status == RoomRentCalculationStatus.INSUFFICIENT_INPUT


def test_missing_policy_cap_is_insufficient_input() -> None:
    result = calculate_room_rent_proportionate_deduction(
        policy_room_rent_cap=None,
        policy_room_rent_cap_unit=None,
        proportionate_deduction_applicable=True,
        actual_room_rent_per_day=6000,
        billed_room_rent_amount=6000,
    )

    assert result.status == RoomRentCalculationStatus.INSUFFICIENT_INPUT
    assert result.room_rent_adjustment is None


def test_missing_actual_room_rent_is_insufficient_input() -> None:
    result = calculate_room_rent_proportionate_deduction(
        policy_room_rent_cap=5000,
        policy_room_rent_cap_unit="INR_per_day",
        proportionate_deduction_applicable=True,
        actual_room_rent_per_day=None,
        billed_room_rent_amount=6000,
    )

    assert result.status == RoomRentCalculationStatus.INSUFFICIENT_INPUT


def test_unsupported_cap_unit_is_insufficient_input() -> None:
    result = calculate_room_rent_proportionate_deduction(
        policy_room_rent_cap=1,
        policy_room_rent_cap_unit="percent_of_sum_insured_per_day",
        proportionate_deduction_applicable=True,
        actual_room_rent_per_day=6000,
        billed_room_rent_amount=6000,
    )

    assert result.status == RoomRentCalculationStatus.INSUFFICIENT_INPUT


def test_calculation_is_deterministically_repeatable() -> None:
    kwargs = dict(
        policy_room_rent_cap=5000,
        policy_room_rent_cap_unit="INR_per_day",
        proportionate_deduction_applicable=True,
        actual_room_rent_per_day=8000,
        billed_room_rent_amount=24000,
        associated_charges_amount=6000,
        source_page=7,
    )

    result_1 = calculate_room_rent_proportionate_deduction(**kwargs)
    result_2 = calculate_room_rent_proportionate_deduction(**kwargs)

    assert result_1.model_dump() == result_2.model_dump()


def _ok_field(value, page: int) -> PolicyField:
    return PolicyField(
        status=FieldStatus.OK,
        value=value,
        confidence=0.9,
        sources=[
            SourceEvidence(
                document_id="doc1", filename="policy.pdf", page_number=page, chunk_id="c1", text="..."
            )
        ],
    )


def test_apply_room_rent_rule_uses_extracted_fields_and_their_page() -> None:
    fields = ExtractedPolicyFields(
        room_rent_cap=_ok_field(5000, page=2),
        room_rent_cap_unit=_ok_field("INR_per_day", page=2),
        proportionate_deduction_applicable=_ok_field(True, page=3),
        copay_percentage=PolicyField.unknown(),
        deductible=PolicyField.unknown(),
        relevant_sub_limits=PolicyField.unknown(),
        potentially_relevant_exclusions=PolicyField.unknown(),
    )

    result = apply_room_rent_rule(
        fields,
        actual_room_rent_per_day=10000,
        billed_room_rent_amount=10000,
        associated_charges_amount=4000,
    )

    assert result.status == RoomRentCalculationStatus.OK
    assert result.source_page == 2  # taken from room_rent_cap's own source, not proportionate flag's
    assert result.room_rent_adjustment == 5000.0
    assert result.associated_charge_adjustment == 2000.0


def test_apply_room_rent_rule_with_unknown_cap_is_insufficient_input() -> None:
    fields = ExtractedPolicyFields(
        room_rent_cap=PolicyField.unknown(),
        room_rent_cap_unit=PolicyField.unknown(),
        proportionate_deduction_applicable=PolicyField.unknown(),
        copay_percentage=PolicyField.unknown(),
        deductible=PolicyField.unknown(),
        relevant_sub_limits=PolicyField.unknown(),
        potentially_relevant_exclusions=PolicyField.unknown(),
    )

    result = apply_room_rent_rule(
        fields,
        actual_room_rent_per_day=10000,
        billed_room_rent_amount=10000,
    )

    assert result.status == RoomRentCalculationStatus.INSUFFICIENT_INPUT
