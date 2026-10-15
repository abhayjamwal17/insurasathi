"""Phase 3 demonstration: retrieval evidence -> LLM interpretation -> structured JSON.

Runs the full Evidence -> Interpretation -> Structured field pipeline on the
same SYNTHETIC test policy used in rag_demo.py.

By default this uses a SIMULATED provider (SimulatedLLMProvider below), so
the demo runs offline and deterministically — this sandbox has no
GROQ_API_KEY set and no local Ollama server running. The simulated
provider stands in for `extraction.providers.LLMProvider` exactly the way
GroqProvider/OllamaProvider do; swapping it for a real one is a one-line
change (see `--live` below).

To run against a REAL provider instead:
    LLM_PROVIDER=groq GROQ_API_KEY=... python extraction_demo.py --live
    LLM_PROVIDER=ollama python extraction_demo.py --live   # needs `ollama serve`
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import fitz

from config import settings
from extraction.extractor import PolicyExtractor
from extraction.providers import LLMProvider, StaticProvider, get_provider
from extraction.schemas import FIELD_NAMES
from ingestion.chunker import chunk_pages
from ingestion.pdf_parser import parse_pdf
from rag.retriever import Retriever
from rag.schemas import RetrievedChunk
from rag.vector_store import ChunkVectorStore

# Clearly-labeled synthetic content — NOT a real insurance policy.
SYNTHETIC_POLICY_PAGES = [
    "SYNTHETIC TEST POLICY DOCUMENT\n"
    "This document is fabricated content used only to demonstrate the "
    "extraction pipeline. It does not represent a real insurance product.",
    "ROOM RENT LIMIT\n"
    "The Company shall pay for room, boarding and nursing expenses as "
    "provided by the hospital up to a maximum of one percent of the Sum "
    "Insured per day, subject to a cap of INR 5000 per day for a standard "
    "private room. Charges above this limit are the insured person's own "
    "responsibility.",
    "INTENSIVE CARE UNIT CHARGES\n"
    "Expenses incurred for Intensive Care Unit admission are payable up to "
    "two percent of the Sum Insured per day, with no fixed rupee cap.",
    "CO-PAYMENT FOR SENIOR CITIZENS\n"
    "A mandatory co-payment of ten percent applies to all admissible "
    "claims for insured persons aged above sixty years at the time of "
    "hospitalization.",
    "PERMANENT EXCLUSIONS\n"
    "The Company shall not be liable for any claim arising from cosmetic "
    "or aesthetic treatment, dental treatment unless arising from an "
    "accident, and treatment for obesity or weight control.",
]


class SimulatedLLMProvider(LLMProvider):
    """Stands in for a real LLM for this offline demo.

    Behaves like a careful model following the extraction prompt's rules:
    it only ever reports a value for a field when the evidence chunk it was
    handed actually states one, and returns "unknown" (never a guess) for
    everything else. This is NOT a hardcoded final answer — it inspects
    whatever evidence chunks are actually passed to it at call time, the
    same way a real LLM would read the prompt's evidence block.
    """

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        # Recover which chunk_ids/pages/text were actually supplied, the
        # same information a real LLM would read out of the prompt text.
        fields = {name: _unknown_field() for name in FIELD_NAMES}

        for line in user_prompt.splitlines():
            if not line.startswith("[chunk_id="):
                continue
            header, _, _ = line.partition("]")
            chunk_id = header.split("chunk_id=")[1].split(" |")[0]
            text_start = user_prompt.find(line) + len(line) + 1
            text = user_prompt[text_start : user_prompt.find("\n\n", text_start)]

            lowered = text.lower()
            if "room rent" in lowered and "5000" in text:
                fields["room_rent_cap"] = _ok_field(5000, [chunk_id], 0.95)
                fields["room_rent_cap_unit"] = _ok_field("INR_per_day", [chunk_id], 0.95)
            if "co-payment" in lowered and "ten percent" in lowered:
                fields["copay_percentage"] = _ok_field(10, [chunk_id], 0.9)
            if "intensive care unit" in lowered:
                fields["relevant_sub_limits"] = _ok_field(
                    ["ICU: 2% of Sum Insured per day, no fixed rupee cap"], [chunk_id], 0.85
                )
            if "exclusions" in lowered or "shall not be liable" in lowered:
                fields["potentially_relevant_exclusions"] = _ok_field(
                    ["Cosmetic/aesthetic treatment", "Dental treatment (unless accident-related)",
                     "Obesity/weight control treatment"],
                    [chunk_id],
                    0.9,
                )

        return json.dumps(fields)


def _unknown_field() -> dict:
    return {
        "status": "unknown",
        "value": None,
        "confidence": None,
        "source_chunk_ids": [],
        "notes": "No supplied evidence establishes this field.",
    }


def _ok_field(value, chunk_ids: list[str], confidence: float) -> dict:
    return {
        "status": "ok",
        "value": value,
        "confidence": confidence,
        "source_chunk_ids": chunk_ids,
        "notes": None,
    }


def _build_synthetic_policy_pdf(path: Path) -> None:
    doc = fitz.open()
    for text in SYNTHETIC_POLICY_PAGES:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 500, 750), text)
    doc.save(path)
    doc.close()


def _print_field(name: str, field) -> None:
    print(f"  {name}: status={field.status.value}, value={field.value!r}, "
          f"confidence={field.confidence}")
    for source in field.sources:
        preview = " ".join(source.text.split())[:90]
        print(f"      <- page {source.page_number} ({source.filename}, "
              f"chunk_id={source.chunk_id}): {preview}...")
    if field.notes:
        print(f"      note: {field.notes}")


def main() -> None:
    live = "--live" in sys.argv

    if live:
        provider = get_provider(
            settings.llm_provider,
            groq_api_key=settings.groq_api_key,
            groq_model=settings.groq_model,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
        )
        print(f"Using LIVE provider: {settings.llm_provider}\n")
    else:
        provider = SimulatedLLMProvider()
        print("Using SIMULATED provider (no network / API key required).")
        print("Pass --live with LLM_PROVIDER + credentials set to use a real LLM.\n")

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = Path(tmp_dir) / "synthetic_policy.pdf"
        _build_synthetic_policy_pdf(pdf_path)

        pages = parse_pdf(pdf_path, document_id="synthetic-policy-demo")
        chunks = chunk_pages(pages)

        store = ChunkVectorStore(collection_name="extraction_demo")
        store.add_chunks(chunks)
        retriever = Retriever(store, top_k=3, relevance_threshold=0.3)
        extractor = PolicyExtractor(provider)

        print("=" * 70)
        print("EXAMPLE 1: clear evidence -> extracted value with page citation")
        print("=" * 70)
        query = "What is the room rent limit and co-payment for this policy?"
        retrieval = retriever.retrieve(query)
        print(f"Query: {query!r}")
        print(f"Retrieved {len(retrieval.chunks)} chunk(s):")
        for c in retrieval.chunks:
            print(f"  - page {c.page_number} (score={c.relevance_score:.3f}): "
                  f"{' '.join(c.text.split())[:70]}...")
        print()

        extracted = extractor.extract(retrieval.chunks)
        print("Extracted structured fields:")
        for name in FIELD_NAMES:
            _print_field(name, getattr(extracted, name))
        print()

        print("=" * 70)
        print("EXAMPLE 2: model correctly refuses to infer missing information")
        print("=" * 70)
        query2 = "What is the deductible for this policy?"
        # Deliberately retrieve evidence that does NOT mention a deductible,
        # to show the extractor returns "unknown" instead of guessing a
        # plausible-sounding number.
        retrieval2 = retriever.retrieve("room rent limit", relevance_threshold=0.3)
        print(f"Query: {query2!r} (evidence supplied is the room-rent clause, "
              "which never mentions a deductible)")
        extracted2 = extractor.extract(retrieval2.chunks)
        _print_field("deductible", extracted2.deductible)
        assert extracted2.deductible.status.value == "unknown"
        assert extracted2.deductible.value is None
        print("\n  -> Correctly abstained: no deductible value was invented.")


if __name__ == "__main__":
    main()
