"""Orchestrates evidence + extracted value + rule + calculation into one
explainable AuditResult, and enforces the system's abstention rules.

The audit layer refuses to calculate — returning NEEDS_REVIEW or
CANNOT_DETERMINE with an explicit reason, never a silently-assumed value —
whenever:

  1. Required policy evidence is missing        -> CANNOT_DETERMINE
  2. Relevant policy clauses conflict            -> NEEDS_REVIEW
  3. A required extracted value is unknown       -> CANNOT_DETERMINE
  4. Bill extraction is insufficient              -> CANNOT_DETERMINE
  5. The applicable rule can't be established     -> CANNOT_DETERMINE

A low-confidence (but present) extraction does not block calculation, but
downgrades the result to NEEDS_REVIEW / AMBER rather than reporting it as
verified.
"""

from __future__ import annotations

from calculation.room_rent import apply_room_rent_rule
from calculation.schemas import RoomRentCalculationStatus
from extraction.bill_schemas import ExtractedBill
from extraction.schemas import ExtractedPolicyFields, FieldStatus, PolicyField
from evidence.schemas import AuditResult, ConfidenceLevel, EvidenceItem, ReviewStatus

ROOM_RENT_RULE_NAME = "room_rent_cap_proportionate_deduction"
LOW_CONFIDENCE_THRESHOLD = 0.5


def _evidence_from_field(field: PolicyField, field_name: str, rule_applied: str) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            document=source.filename,
            page=source.page_number,
            source_text=source.text,
            extracted_field=field_name,
            confidence=field.confidence,
            rule_applied=rule_applied,
        )
        for source in field.sources
    ]


def _cannot_determine(reason: str, evidence: list[EvidenceItem] | None = None) -> AuditResult:
    return AuditResult(
        status=ReviewStatus.CANNOT_DETERMINE,
        confidence_level=ConfidenceLevel.RED,
        evidence=evidence or [],
        reason=reason,
    )


def _needs_review(reason: str, evidence: list[EvidenceItem] | None = None) -> AuditResult:
    return AuditResult(
        status=ReviewStatus.NEEDS_REVIEW,
        confidence_level=ConfidenceLevel.AMBER,
        evidence=evidence or [],
        reason=reason,
    )


def audit_room_rent_charge(
    policy_fields: ExtractedPolicyFields,
    bill: ExtractedBill,
    actual_room_rent_per_day: float | None,
) -> AuditResult:
    """Run the room-rent-cap audit end to end, refusing to guess anywhere.

    actual_room_rent_per_day is an explicit input (not inferred from the
    bill) for the same reason calculation.room_rent requires it explicitly:
    turning a billed total into a daily rate requires knowing the number of
    days, which this layer will not assume.
    """
    cap_field = policy_fields.room_rent_cap
    unit_field = policy_fields.room_rent_cap_unit
    prop_field = policy_fields.proportionate_deduction_applicable

    # --- Abstention rule 1 & 3: required policy evidence missing/unknown ---
    for name, field in (("room_rent_cap", cap_field), ("room_rent_cap_unit", unit_field)):
        if field.status == FieldStatus.UNKNOWN:
            return _cannot_determine(
                f"Required policy evidence for '{name}' is missing "
                f"(extraction status: unknown; not established by any retrieved clause)."
            )

    # --- Abstention rule 2: relevant clauses conflict ---
    for name, field in (("room_rent_cap", cap_field), ("room_rent_cap_unit", unit_field)):
        if field.status == FieldStatus.NEEDS_REVIEW:
            return _needs_review(
                f"Relevant policy clauses conflict for '{name}': "
                f"{field.notes or 'multiple sources state different values.'}",
                evidence=_evidence_from_field(field, name, ROOM_RENT_RULE_NAME),
            )

    # --- Abstention rule 4: bill extraction insufficient ---
    room_rent_items = [item for item in bill.line_items if item.category == "room_rent"]
    if not room_rent_items:
        return _cannot_determine(
            "Bill extraction is insufficient: no line item was categorized as 'room_rent'."
        )
    if any(item.amount is None for item in room_rent_items):
        return _cannot_determine(
            "Bill extraction is insufficient: a room_rent line item has no extracted amount."
        )
    if actual_room_rent_per_day is None or actual_room_rent_per_day <= 0:
        return _cannot_determine(
            "Bill extraction is insufficient: actual room rent per day could not be determined."
        )

    billed_room_rent_amount = sum(item.amount for item in room_rent_items)  # type: ignore[misc]

    # --- Run the deterministic calculation ---
    calc_result = apply_room_rent_rule(
        policy_fields,
        actual_room_rent_per_day=actual_room_rent_per_day,
        billed_room_rent_amount=billed_room_rent_amount,
    )

    # --- Abstention rule 5: the applicable rule cannot be established ---
    if calc_result.status == RoomRentCalculationStatus.INSUFFICIENT_INPUT:
        return _cannot_determine(
            f"The applicable rule cannot be established from evidence: {calc_result.notes}"
        )

    evidence: list[EvidenceItem] = []
    for name, field in (
        ("room_rent_cap", cap_field),
        ("room_rent_cap_unit", unit_field),
        ("proportionate_deduction_applicable", prop_field),
    ):
        evidence.extend(_evidence_from_field(field, name, ROOM_RENT_RULE_NAME))

    warnings: list[str] = []
    unknown_items = [item for item in bill.line_items if item.category == "unknown"]
    if unknown_items:
        total_unknown = sum(item.amount or 0 for item in unknown_items)
        warnings.append(
            f"Bill contains {len(unknown_items)} line item(s) categorized 'unknown' "
            f"(total INR {total_unknown:.2f}) that were not evaluated by any rule."
        )

    low_confidence_fields = [
        name
        for name, field in (("room_rent_cap", cap_field), ("room_rent_cap_unit", unit_field))
        if field.confidence is not None and field.confidence < LOW_CONFIDENCE_THRESHOLD
    ]
    if low_confidence_fields:
        warnings.append(
            f"Extraction confidence below {LOW_CONFIDENCE_THRESHOLD} for: {low_confidence_fields}."
        )
        return AuditResult(
            status=ReviewStatus.NEEDS_REVIEW,
            confidence_level=ConfidenceLevel.AMBER,
            estimated_result=calc_result,
            applied_rules=[ROOM_RENT_RULE_NAME],
            evidence=evidence,
            warnings=warnings,
            reason=f"Low extraction confidence for: {low_confidence_fields}.",
        )

    if calc_result.status == RoomRentCalculationStatus.NEEDS_REVIEW:
        warnings.append(
            "Whether proportionate deduction extends to associated charges is not "
            "established by the evidence."
        )
        return AuditResult(
            status=ReviewStatus.NEEDS_REVIEW,
            confidence_level=ConfidenceLevel.AMBER,
            estimated_result=calc_result,
            applied_rules=[ROOM_RENT_RULE_NAME],
            evidence=evidence,
            warnings=warnings,
            reason="Calculation was completed for room rent itself, but associated-charge "
            "applicability is unresolved; see calculation trace.",
        )

    return AuditResult(
        status=ReviewStatus.OK,
        confidence_level=ConfidenceLevel.GREEN,
        estimated_result=calc_result,
        applied_rules=[ROOM_RENT_RULE_NAME],
        evidence=evidence,
        warnings=warnings,
        reason=None,
    )
