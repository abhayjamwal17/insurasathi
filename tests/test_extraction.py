"""Tests for the Phase 3 LLM extraction layer.

Uses StaticProvider (extraction/providers.py) to exercise the extraction
pipeline deterministically, without a network call or a running LLM. What's
under test is the extractor's own logic: prompt/response wiring, JSON
parsing, unknown/needs_review handling, and — critically — that page/
document provenance is attached from the *supplied* evidence chunks, not
from anything the LLM claims.
"""

from __future__ import annotations

import json

import pytest

from extraction.extractor import ExtractionError, PolicyExtractor
from extraction.providers import LLMProvider, StaticProvider
from extraction.schemas import FIELD_NAMES, FieldStatus
from rag.schemas import RetrievedChunk


def _chunk(chunk_id: str, page_number: int, text: str, filename: str = "policy.pdf") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-test-001",
        filename=filename,
        page_number=page_number,
        text=text,
        relevance_score=0.9,
    )


def _default_field() -> dict:
    return {
        "status": "unknown",
        "value": None,
        "confidence": None,
        "source_chunk_ids": [],
        "notes": "No evidence supplied for this field.",
    }


def _response(overrides: dict[str, dict]) -> str:
    """Build a full, schema-complete static LLM response as a JSON string."""
    fields = {name: _default_field() for name in FIELD_NAMES}
    for name, override in overrides.items():
        fields[name] = override
    return json.dumps(fields)


class CountingProvider(LLMProvider):
    """Wraps another provider and records how many times it was called."""

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.call_count = 0

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        return self._inner.complete_json(system_prompt, user_prompt)


def test_clear_room_rent_clause_is_extracted_with_page_citation() -> None:
    chunk = _chunk("c1", page_number=2, text="ROOM RENT LIMIT: capped at INR 5000 per day.")
    response = _response(
        {
            "room_rent_cap": {
                "status": "ok",
                "value": 5000,
                "confidence": 0.95,
                "source_chunk_ids": ["c1"],
                "notes": None,
            },
            "room_rent_cap_unit": {
                "status": "ok",
                "value": "INR_per_day",
                "confidence": 0.95,
                "source_chunk_ids": ["c1"],
                "notes": None,
            },
        }
    )
    extractor = PolicyExtractor(StaticProvider(response))

    result = extractor.extract([chunk])

    assert result.room_rent_cap.status == FieldStatus.OK
    assert result.room_rent_cap.value == 5000
    assert len(result.room_rent_cap.sources) == 1
    source = result.room_rent_cap.sources[0]
    assert source.page_number == 2
    assert source.chunk_id == "c1"
    assert source.filename == "policy.pdf"

    assert result.room_rent_cap_unit.value == "INR_per_day"


def test_clear_copay_clause_is_extracted_with_page_citation() -> None:
    chunk = _chunk("c2", page_number=4, text="A mandatory co-payment of 10% applies to seniors.")
    response = _response(
        {
            "copay_percentage": {
                "status": "ok",
                "value": 10,
                "confidence": 0.9,
                "source_chunk_ids": ["c2"],
                "notes": None,
            }
        }
    )
    extractor = PolicyExtractor(StaticProvider(response))

    result = extractor.extract([chunk])

    assert result.copay_percentage.status == FieldStatus.OK
    assert result.copay_percentage.value == 10
    assert result.copay_percentage.sources[0].page_number == 4


def test_missing_information_returns_unknown_not_a_guess() -> None:
    chunk = _chunk("c3", page_number=1, text="This policy covers hospitalization expenses.")
    response = _response({})  # everything defaults to unknown
    extractor = PolicyExtractor(StaticProvider(response))

    result = extractor.extract([chunk])

    assert result.deductible.status == FieldStatus.UNKNOWN
    assert result.deductible.value is None
    assert result.deductible.confidence is None
    assert result.deductible.sources == []


def test_conflicting_evidence_produces_needs_review_not_a_choice() -> None:
    chunk_a = _chunk("c4", page_number=3, text="Room rent is capped at INR 5000 per day.")
    chunk_b = _chunk("c5", page_number=9, text="Endorsement: room rent capped at INR 7500 per day.")
    response = _response(
        {
            "room_rent_cap": {
                "status": "needs_review",
                "value": None,
                "confidence": None,
                "source_chunk_ids": ["c4", "c5"],
                "notes": "Base policy states INR 5000/day; endorsement states INR 7500/day.",
            }
        }
    )
    extractor = PolicyExtractor(StaticProvider(response))

    result = extractor.extract([chunk_a, chunk_b])

    assert result.room_rent_cap.status == FieldStatus.NEEDS_REVIEW
    assert result.room_rent_cap.value is None
    assert {s.chunk_id for s in result.room_rent_cap.sources} == {"c4", "c5"}
    assert {s.page_number for s in result.room_rent_cap.sources} == {3, 9}
    assert result.room_rent_cap.notes is not None


def test_page_metadata_is_taken_from_supplied_evidence_not_llm_claims() -> None:
    chunk = _chunk("c6", page_number=14, text="ICU charges: 2% of Sum Insured per day.")
    response = _response(
        {
            "relevant_sub_limits": {
                "status": "ok",
                "value": ["ICU: 2% of Sum Insured per day"],
                "confidence": 0.8,
                # LLM claims a bogus page number inline in notes; must be ignored.
                "source_chunk_ids": ["c6"],
                "notes": "Stated on page 999 according to the model (should be ignored).",
            }
        }
    )
    extractor = PolicyExtractor(StaticProvider(response))

    result = extractor.extract([chunk])

    assert result.relevant_sub_limits.sources[0].page_number == 14


def test_unrecognized_chunk_id_is_dropped_not_trusted() -> None:
    chunk = _chunk("c7", page_number=5, text="Some clause text.")
    response = _response(
        {
            "deductible": {
                "status": "ok",
                "value": 1000,
                "confidence": 0.7,
                "source_chunk_ids": ["c7", "does-not-exist"],
                "notes": None,
            }
        }
    )
    extractor = PolicyExtractor(StaticProvider(response))

    result = extractor.extract([chunk])

    assert [s.chunk_id for s in result.deductible.sources] == ["c7"]


def test_empty_evidence_short_circuits_without_calling_the_llm() -> None:
    inner = StaticProvider(_response({}))
    provider = CountingProvider(inner)
    extractor = PolicyExtractor(provider)

    result = extractor.extract([])

    assert provider.call_count == 0
    assert all(getattr(result, name).status == FieldStatus.UNKNOWN for name in FIELD_NAMES)


def test_malformed_json_raises_extraction_error() -> None:
    chunk = _chunk("c8", page_number=1, text="Some clause text.")
    extractor = PolicyExtractor(StaticProvider("this is not json"))

    with pytest.raises(ExtractionError):
        extractor.extract([chunk])


def test_missing_field_in_llm_output_raises_extraction_error() -> None:
    chunk = _chunk("c9", page_number=1, text="Some clause text.")
    incomplete = json.dumps({"room_rent_cap": _default_field()})
    extractor = PolicyExtractor(StaticProvider(incomplete))

    with pytest.raises(ExtractionError):
        extractor.extract([chunk])
