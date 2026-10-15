"""Room rent cap + proportionate deduction — the first supported rule.

CRITICAL: no LLM calls anywhere in this module. Everything here is plain,
deterministic Python arithmetic over explicit, already-validated numbers.

Rule (only applied when the policy evidence explicitly establishes it):

    ratio = policy_room_rent_cap / actual_room_rent_per_day

    If actual_room_rent_per_day <= policy_room_rent_cap: no deduction.

    Otherwise, the billed room rent itself is reduced proportionately:
        room_rent_adjustment = billed_room_rent_amount * (1 - ratio)

    Whether OTHER ("associated") charges are also reduced proportionately
    depends entirely on the policy's own `proportionate_deduction_applicable`
    fact:
      - True  -> the caller-supplied associated_charges_amount is reduced
                 by the same ratio.
      - False -> no other charges are touched.
      - unknown/needs_review -> the room rent adjustment still applies (the
        cap itself is a fact), but the associated-charges question is left
        unresolved and the result is flagged "needs_review" rather than the
        engine guessing an answer.

    This module never decides *which* bill categories count as "associated
    charges" — that scope must come from policy evidence and is passed in
    by the caller as associated_charges_amount. If evidence hasn't
    established it, callers should pass 0.0, and no charges beyond room
    rent are touched.
"""

from __future__ import annotations

from calculation.schemas import CalculationTrace, RoomRentCalculationResult, RoomRentCalculationStatus
from extraction.schemas import ExtractedPolicyFields, FieldStatus

SUPPORTED_ROOM_RENT_CAP_UNIT = "INR_per_day"


def _insufficient(trace: CalculationTrace, notes: str) -> RoomRentCalculationResult:
    return RoomRentCalculationResult(
        status=RoomRentCalculationStatus.INSUFFICIENT_INPUT,
        trace=trace,
        notes=notes,
    )


def calculate_room_rent_proportionate_deduction(
    *,
    policy_room_rent_cap: float | None,
    policy_room_rent_cap_unit: str | None,
    proportionate_deduction_applicable: bool | None,
    actual_room_rent_per_day: float | None,
    billed_room_rent_amount: float,
    associated_charges_amount: float = 0.0,
    source_page: int | None = None,
) -> RoomRentCalculationResult:
    """Pure, deterministic room-rent cap + proportionate deduction calculation.

    `proportionate_deduction_applicable=None` means the policy evidence did
    not establish whether the deduction extends to associated charges
    (extraction status was "unknown" or "needs_review") — this function
    does not guess in that case; see module docstring.
    """
    trace = CalculationTrace()

    if policy_room_rent_cap is None or policy_room_rent_cap_unit is None:
        trace.add(
            "Missing policy input",
            "room_rent_cap or room_rent_cap_unit was not established by policy evidence.",
        )
        return _insufficient(trace, "Policy room rent cap is not established by the evidence.")

    if policy_room_rent_cap_unit != SUPPORTED_ROOM_RENT_CAP_UNIT:
        trace.add(
            "Unsupported unit",
            f"room_rent_cap_unit={policy_room_rent_cap_unit!r}; this rule only supports "
            f"{SUPPORTED_ROOM_RENT_CAP_UNIT!r}.",
        )
        return _insufficient(trace, f"Unsupported room_rent_cap_unit: {policy_room_rent_cap_unit!r}")

    if policy_room_rent_cap <= 0:
        trace.add("Invalid policy input", f"policy_room_rent_cap={policy_room_rent_cap} must be > 0.")
        return _insufficient(trace, "policy_room_rent_cap must be a positive number.")

    if actual_room_rent_per_day is None:
        trace.add("Missing bill input", "actual_room_rent_per_day was not supplied.")
        return _insufficient(trace, "actual_room_rent_per_day is required.")

    if actual_room_rent_per_day <= 0:
        trace.add("Invalid bill input", f"actual_room_rent_per_day={actual_room_rent_per_day} must be > 0.")
        return _insufficient(trace, "actual_room_rent_per_day must be a positive number.")

    trace.add(
        "Input",
        f"policy_room_rent_cap = INR {policy_room_rent_cap:.2f}/day, "
        f"actual_room_rent_per_day = INR {actual_room_rent_per_day:.2f}/day",
    )

    if actual_room_rent_per_day <= policy_room_rent_cap:
        trace.add(
            "Comparison",
            "actual_room_rent_per_day <= policy_room_rent_cap: no proportionate deduction applies.",
        )
        return RoomRentCalculationResult(
            status=RoomRentCalculationStatus.NOT_APPLICABLE,
            policy_room_rent_cap=policy_room_rent_cap,
            policy_room_rent_cap_unit=policy_room_rent_cap_unit,
            actual_room_rent_per_day=actual_room_rent_per_day,
            ratio=1.0,
            room_rent_adjustment=0.0,
            associated_charge_adjustment=0.0,
            total_adjustment=0.0,
            final_room_rent_amount=round(billed_room_rent_amount, 2),
            source_page=source_page,
            trace=trace,
        )

    ratio = policy_room_rent_cap / actual_room_rent_per_day
    trace.add("Formula", "ratio = policy_room_rent_cap / actual_room_rent_per_day")
    trace.add("Ratio", f"{ratio:.4f} ({ratio * 100:.2f}%)")

    room_rent_adjustment = billed_room_rent_amount * (1 - ratio)
    final_room_rent_amount = billed_room_rent_amount - room_rent_adjustment
    trace.add(
        "Room rent adjustment",
        f"billed_room_rent_amount * (1 - ratio) = "
        f"{billed_room_rent_amount:.2f} * (1 - {ratio:.4f}) = INR {room_rent_adjustment:.2f}",
    )

    if proportionate_deduction_applicable is True:
        associated_adjustment = associated_charges_amount * (1 - ratio)
        trace.add(
            "Associated charges",
            "Policy evidence establishes proportionate deduction applies to associated "
            f"charges: {associated_charges_amount:.2f} * (1 - {ratio:.4f}) = "
            f"INR {associated_adjustment:.2f}",
        )
        status = RoomRentCalculationStatus.OK
    elif proportionate_deduction_applicable is False:
        associated_adjustment = 0.0
        trace.add(
            "Associated charges",
            "Policy evidence establishes proportionate deduction does NOT extend beyond "
            "room rent itself; no associated charges adjusted.",
        )
        status = RoomRentCalculationStatus.OK
    else:
        associated_adjustment = 0.0
        trace.add(
            "Associated charges",
            "Whether proportionate deduction extends to associated charges is not "
            "established by the evidence (unknown/needs_review). Only the room rent "
            "line item has been adjusted; associated charges are left for human review.",
        )
        status = RoomRentCalculationStatus.NEEDS_REVIEW

    total_adjustment = room_rent_adjustment + associated_adjustment
    trace.add(
        "Final amount",
        f"total_adjustment = room_rent_adjustment + associated_charge_adjustment = "
        f"INR {total_adjustment:.2f}",
    )
    if source_page is not None:
        trace.add("Source policy page", f"page {source_page}")

    return RoomRentCalculationResult(
        status=status,
        policy_room_rent_cap=policy_room_rent_cap,
        policy_room_rent_cap_unit=policy_room_rent_cap_unit,
        actual_room_rent_per_day=actual_room_rent_per_day,
        ratio=ratio,
        room_rent_adjustment=round(room_rent_adjustment, 2),
        associated_charge_adjustment=round(associated_adjustment, 2),
        total_adjustment=round(total_adjustment, 2),
        final_room_rent_amount=round(final_room_rent_amount, 2),
        source_page=source_page,
        trace=trace,
    )


# --- Extracted fact -> policy rule input -----------------------------------
# These helpers are the explicit boundary between "what the LLM extracted"
# (extraction.schemas.PolicyField, with its own status) and "what this rule
# is allowed to treat as a known fact." A field that isn't status == "ok" is
# never coerced into a value here — it becomes None, and the calculation
# function above turns that into INSUFFICIENT_INPUT/NEEDS_REVIEW rather than
# silently treating "unknown" as if it were a fact.


def room_rent_cap_from_extraction(fields: ExtractedPolicyFields) -> tuple[float | None, str | None]:
    """Pull (cap, unit) from extracted fields, only if both are status='ok'."""
    cap_field = fields.room_rent_cap
    unit_field = fields.room_rent_cap_unit
    if cap_field.status != FieldStatus.OK or unit_field.status != FieldStatus.OK:
        return None, None
    if not isinstance(cap_field.value, (int, float)) or not isinstance(unit_field.value, str):
        return None, None
    return float(cap_field.value), unit_field.value


def room_rent_cap_source_page(fields: ExtractedPolicyFields) -> int | None:
    cap_field = fields.room_rent_cap
    if cap_field.status != FieldStatus.OK or not cap_field.sources:
        return None
    return cap_field.sources[0].page_number


def proportionate_deduction_flag_from_extraction(fields: ExtractedPolicyFields) -> bool | None:
    """Returns True/False only if the policy evidence establishes it; else None."""
    field = fields.proportionate_deduction_applicable
    if field.status != FieldStatus.OK or not isinstance(field.value, bool):
        return None
    return field.value


def apply_room_rent_rule(
    policy_fields: ExtractedPolicyFields,
    *,
    actual_room_rent_per_day: float | None,
    billed_room_rent_amount: float,
    associated_charges_amount: float = 0.0,
) -> RoomRentCalculationResult:
    """Orchestration layer: extracted fact -> policy rule -> calculation -> result.

    Translates ExtractedPolicyFields into plain calculation inputs, then
    delegates to the pure calculate_room_rent_proportionate_deduction().
    """
    cap, unit = room_rent_cap_from_extraction(policy_fields)
    proportionate_flag = proportionate_deduction_flag_from_extraction(policy_fields)
    source_page = room_rent_cap_source_page(policy_fields)

    return calculate_room_rent_proportionate_deduction(
        policy_room_rent_cap=cap,
        policy_room_rent_cap_unit=unit,
        proportionate_deduction_applicable=proportionate_flag,
        actual_room_rent_per_day=actual_room_rent_per_day,
        billed_room_rent_amount=billed_room_rent_amount,
        associated_charges_amount=associated_charges_amount,
        source_page=source_page,
    )
