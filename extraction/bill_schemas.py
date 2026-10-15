"""Structured hospital interim bill line items.

This module only extracts and categorizes what a bill says — it never
decides whether a line item is covered by any policy. Coverage decisions,
like arithmetic, belong to a later, deterministic phase.

Categories are a controlled but extensible list (BILL_LINE_ITEM_CATEGORIES)
rather than a closed Enum, since real hospital bills use inconsistent
terminology and new categories may need to be added over time without
changing every call site. An item is categorized "unknown" whenever the
evidence doesn't confidently support a more specific category — the
extractor must never guess.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

BILL_LINE_ITEM_CATEGORIES: tuple[str, ...] = (
    "room_rent",
    "nursing",
    "consultant_doctor_charges",
    "ot_charges",
    "medicines",
    "consumables",
    "administrative_charges",
    "other_hospital_charges",
    "unknown",
)

UNKNOWN_CATEGORY = "unknown"


class BillLineItem(BaseModel):
    """One billed line item, categorized but not yet judged for coverage."""

    description: str
    amount: float | None = Field(
        default=None, description="Line total in the bill's currency. Null if not stated."
    )
    quantity: float | None = Field(default=None, description="Null if not stated.")
    unit_price: float | None = Field(default=None, description="Null if not stated.")
    category: str = Field(
        ..., description=f"One of {BILL_LINE_ITEM_CATEGORIES}; 'unknown' if not confidently determined."
    )
    page_number: int
    source_text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("category")
    @classmethod
    def _category_must_be_controlled(cls, value: str) -> str:
        if value not in BILL_LINE_ITEM_CATEGORIES:
            raise ValueError(
                f"category {value!r} is not in the controlled list {BILL_LINE_ITEM_CATEGORIES}"
            )
        return value


class ExtractedBill(BaseModel):
    """A normalized, page-aware representation of one hospital bill."""

    document_id: str
    filename: str
    line_items: list[BillLineItem] = Field(default_factory=list)
