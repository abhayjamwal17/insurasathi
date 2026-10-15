# Deterministic financial calculation engine.
# CRITICAL: this layer performs all arithmetic. No LLM calls happen here —
# it only consumes already-validated structured inputs (Pydantic models
# from extraction/) and explicit rule definitions.

from calculation.room_rent import (
    apply_room_rent_rule,
    calculate_room_rent_proportionate_deduction,
    proportionate_deduction_flag_from_extraction,
    room_rent_cap_from_extraction,
    room_rent_cap_source_page,
)
from calculation.schemas import (
    CalculationTrace,
    RoomRentCalculationResult,
    RoomRentCalculationStatus,
    TraceStep,
)

__all__ = [
    "calculate_room_rent_proportionate_deduction",
    "apply_room_rent_rule",
    "room_rent_cap_from_extraction",
    "room_rent_cap_source_page",
    "proportionate_deduction_flag_from_extraction",
    "CalculationTrace",
    "TraceStep",
    "RoomRentCalculationResult",
    "RoomRentCalculationStatus",
]
