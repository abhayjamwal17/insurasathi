"""Structured policy schema produced by LLM interpretation.

Every extracted field carries its own status and provenance rather than a
bare value: the LLM interprets evidence, but the calling code (and later,
a human reviewer) must always be able to see *why* a field has the value
it has, or *why* it doesn't have one.

CRITICAL: nothing in this module performs arithmetic. Values here are raw
interpreted facts (a cap, a percentage, a list of exclusions) — applying
them to a bill happens later, in deterministic Python, never here.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

FieldValue = float | int | str | bool | list[str] | None


class FieldStatus(str, Enum):
    """How a field's value was determined."""

    OK = "ok"
    UNKNOWN = "unknown"
    NEEDS_REVIEW = "needs_review"


class SourceEvidence(BaseModel):
    """The exact source chunk a field's value (or conflict) was drawn from.

    Populated by the extractor from the retrieved chunks that were actually
    supplied to the LLM — never from the LLM's own (unverified) claims about
    page numbers, so provenance can't be hallucinated.
    """

    document_id: str
    filename: str
    page_number: int
    chunk_id: str
    text: str


class PolicyField(BaseModel):
    """One interpreted field: a value (maybe), a status, and its evidence."""

    status: FieldStatus
    value: FieldValue = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    sources: list[SourceEvidence] = Field(default_factory=list)
    notes: str | None = Field(
        default=None,
        description="Free-text explanation, required when status is "
        "needs_review (describes the conflict) and useful for unknown.",
    )

    @classmethod
    def unknown(cls, notes: str | None = None) -> "PolicyField":
        return cls(status=FieldStatus.UNKNOWN, value=None, confidence=None, sources=[], notes=notes)


FIELD_NAMES: tuple[str, ...] = (
    "room_rent_cap",
    "room_rent_cap_unit",
    "proportionate_deduction_applicable",
    "copay_percentage",
    "deductible",
    "relevant_sub_limits",
    "potentially_relevant_exclusions",
)

FIELD_DESCRIPTIONS: dict[str, str] = {
    "room_rent_cap": (
        "Numeric cap on room rent, only if the evidence states a specific "
        "number (e.g. a rupee amount, or a percentage of Sum Insured). "
        "Null if no cap is stated."
    ),
    "room_rent_cap_unit": (
        "Unit for room_rent_cap: e.g. 'INR_per_day' or "
        "'percent_of_sum_insured_per_day'. Null if room_rent_cap is null."
    ),
    "proportionate_deduction_applicable": (
        "true or false: whether the policy text says a proportionate "
        "deduction applies to other expenses when room rent exceeds its "
        "cap. Null if the evidence does not address this."
    ),
    "copay_percentage": (
        "Numeric co-payment percentage the insured must bear, only if a "
        "specific percentage is stated. Null if not stated."
    ),
    "deductible": (
        "Numeric deductible amount, only if a specific amount is stated. "
        "Null if not stated."
    ),
    "relevant_sub_limits": (
        "List of short strings, each describing one other sub-limit found "
        "in the evidence (e.g. 'ICU: 2% of Sum Insured per day'). Empty "
        "list if none are stated."
    ),
    "potentially_relevant_exclusions": (
        "List of short strings, each describing one exclusion found in the "
        "evidence that could be relevant. Empty list if none are stated."
    ),
}


class ExtractedPolicyFields(BaseModel):
    """The full set of structured fields extracted for one query/claim context."""

    room_rent_cap: PolicyField
    room_rent_cap_unit: PolicyField
    proportionate_deduction_applicable: PolicyField
    copay_percentage: PolicyField
    deductible: PolicyField
    relevant_sub_limits: PolicyField
    potentially_relevant_exclusions: PolicyField

    @classmethod
    def all_unknown(cls, notes: str | None = None) -> "ExtractedPolicyFields":
        """Used when there is no evidence at all to extract from."""
        return cls(**{name: PolicyField.unknown(notes=notes) for name in FIELD_NAMES})
