"""Result and trace types for the deterministic calculation engine.

Every calculation produces a CalculationTrace — a plain, ordered list of
labeled steps — so a human reviewer (or a capstone viva panel) can follow
exactly how a number was derived from its inputs, with no hidden logic.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    label: str
    detail: str


class CalculationTrace(BaseModel):
    """An ordered, human-readable record of how a result was derived."""

    steps: list[TraceStep] = Field(default_factory=list)

    def add(self, label: str, detail: str) -> None:
        self.steps.append(TraceStep(label=label, detail=detail))


class RoomRentCalculationStatus(str, Enum):
    """How the room-rent cap / proportionate-deduction rule resolved."""

    OK = "ok"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_INPUT = "insufficient_input"
    NEEDS_REVIEW = "needs_review"


class RoomRentCalculationResult(BaseModel):
    """Deterministic output of the room-rent cap + proportionate deduction rule."""

    status: RoomRentCalculationStatus
    policy_room_rent_cap: float | None = None
    policy_room_rent_cap_unit: str | None = None
    actual_room_rent_per_day: float | None = None
    ratio: float | None = Field(default=None, description="policy_room_rent_cap / actual_room_rent_per_day")
    room_rent_adjustment: float | None = None
    associated_charge_adjustment: float | None = None
    total_adjustment: float | None = None
    final_room_rent_amount: float | None = None
    source_page: int | None = None
    trace: CalculationTrace = Field(default_factory=CalculationTrace)
    notes: str | None = None
