"""Evidence, confidence, and unified audit-result types.

Every important result must be explainable through:
    policy evidence + extracted value + rule + calculation

This module defines the shapes that make that explanation concrete:
EvidenceItem ties one piece of source text to the field it supports and the
rule that consumed it; AuditResult is the single object a caller (or the
eventual UI) reads to learn what happened, whether it's trustworthy, and
whether a human needs to look at it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from calculation.schemas import RoomRentCalculationResult


class ConfidenceLevel(str, Enum):
    """Traffic-light confidence, shown to a human reviewer."""

    GREEN = "GREEN"  # Verified / High confidence
    AMBER = "AMBER"  # Needs review
    RED = "RED"  # Cannot determine / escalation required


class ReviewStatus(str, Enum):
    """Whether the system will stand behind a result unassisted."""

    OK = "OK"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    CANNOT_DETERMINE = "CANNOT_DETERMINE"


class EvidenceItem(BaseModel):
    """One piece of source evidence backing an extracted field / rule."""

    document: str
    page: int
    source_text: str
    extracted_field: str
    confidence: float | None = None
    rule_applied: str | None = None


class AuditResult(BaseModel):
    """The unified, explainable output of one audit run."""

    status: ReviewStatus
    confidence_level: ConfidenceLevel
    estimated_result: RoomRentCalculationResult | None = Field(
        default=None, description="Present only when a calculation was actually performed."
    )
    applied_rules: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reason: str | None = Field(
        default=None,
        description="Required whenever status is NEEDS_REVIEW or CANNOT_DETERMINE.",
    )
