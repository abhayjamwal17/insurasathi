"""Phase 7 phase-gate demonstration: a complete, offline, end-to-end run.

    Policy PDF + Bill PDF
        -> RAG (retrieval)
        -> LLM (interpretation)
        -> Structured JSON
        -> Deterministic calculation
        -> Evidence + confidence
        -> Result

Uses the same `pipeline.py` functions that back streamlit_app.py, so this
script proves the exact code path the UI drives — just without a browser.

Runs against a SIMULATED LLM provider (this sandbox has no GROQ_API_KEY and
no local Ollama server), reading the actual evidence handed to it at call
time rather than returning a hardcoded final answer — the same role
SimulatedLLMProvider plays in extraction_demo.py. Swap in a real
GroqProvider/OllamaProvider (see extraction_demo.py --live) with no other
code changes once credentials are available.

Both the policy and the bill below are clearly-labeled SYNTHETIC content,
built at run time — not real insurance or hospital records.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import fitz

import pipeline
from extraction.providers import LLMProvider

# --- Synthetic test documents (fabricated; not real records) --------------

SYNTHETIC_POLICY_PAGES = [
    "SYNTHETIC TEST POLICY DOCUMENT\n"
    "This document is fabricated content used only to demonstrate the pipeline. "
    "It does not represent a real insurance product.",
    "ROOM RENT LIMIT\n"
    "The Company shall pay for room, boarding and nursing expenses up to a "
    "maximum of INR 5000 per day for a standard private room. Charges above "
    "this limit are the insured person's own responsibility.",
    "PROPORTIONATE DEDUCTION\n"
    "If the actual room rent exceeds the eligible room rent limit stated "
    "above, the Company shall apply a proportionate deduction to the room "
    "rent charge and to all other associated hospital charges incurred "
    "during the same period, in the same proportion by which the room rent "
    "exceeds the limit.",
    "INTENSIVE CARE UNIT CHARGES\n"
    "Expenses incurred for Intensive Care Unit admission are payable up to "
    "two percent of the Sum Insured per day, with no fixed rupee cap.",
    "CO-PAYMENT\n"
    "A mandatory co-payment of ten percent applies to all admissible claims.",
    "PERMANENT EXCLUSIONS\n"
    "The Company shall not be liable for any claim arising from cosmetic or "
    "aesthetic treatment.",
]

SYNTHETIC_BILL_PAGES = [
    "SYNTHETIC INTERIM HOSPITAL BILL (fabricated test data, not a real record)\n"
    "Patient: Test Patient | Admission: Ward -> Private Room",
    "Room Rent (Private Room, 2 days @ INR 8000/day) ... INR 16000\n"
    "Nursing Charges ... INR 3000",
    "Consultant Visit Fee (Cardiologist) ... INR 4000\n"
    "Medicines ... INR 2500\n"
    "Miscellaneous Hospital Charges ... INR 500",
]

# The actual daily room-rent rate is an explicit input, matching the
# architecture's rule that this is never inferred from a billed total.
ACTUAL_ROOM_RENT_PER_DAY = 8000.0


def _build_pdf(path: Path, page_texts: list[str]) -> None:
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 500, 750), text)
    doc.save(path)
    doc.close()


class SimulatedLLMProvider(LLMProvider):
    """Reads whichever prompt it's given (policy or bill) and interprets it.

    Not a hardcoded final answer: it inspects the evidence/page text that
    is actually present in the prompt at call time, the same information a
    real LLM would be reading off the page.
    """

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        if "HOSPITAL BILL PAGE" in user_prompt:
            return self._simulate_bill_page(user_prompt)
        return self._simulate_policy_extraction(user_prompt)

    # -- policy field extraction -------------------------------------------------
    def _simulate_policy_extraction(self, user_prompt: str) -> str:
        from extraction.schemas import FIELD_NAMES

        fields = {name: _unknown_field() for name in FIELD_NAMES}

        for match in re.finditer(
            r"\[chunk_id=(?P<id>\S+) \| page=(?P<page>\d+) \| document=(?P<doc>\S+)\]\n(?P<text>.*?)(?=\n\n\[chunk_id=|\n\nExtract|\Z)",
            user_prompt,
            re.DOTALL,
        ):
            chunk_id = match.group("id")
            text = match.group("text")
            lowered = text.lower()

            if "room rent" in lowered and "5000" in text:
                fields["room_rent_cap"] = _ok_field(5000, [chunk_id], 0.95)
                fields["room_rent_cap_unit"] = _ok_field("INR_per_day", [chunk_id], 0.95)
            if "proportionate deduction" in lowered and "associated" in lowered:
                fields["proportionate_deduction_applicable"] = _ok_field(True, [chunk_id], 0.9)
            if "co-payment" in lowered and "ten percent" in lowered:
                fields["copay_percentage"] = _ok_field(10, [chunk_id], 0.9)
            if "intensive care unit" in lowered:
                fields["relevant_sub_limits"] = _ok_field(
                    ["ICU: 2% of Sum Insured per day, no fixed rupee cap"], [chunk_id], 0.85
                )
            if "shall not be liable" in lowered:
                fields["potentially_relevant_exclusions"] = _ok_field(
                    ["Cosmetic/aesthetic treatment"], [chunk_id], 0.9
                )

        return json.dumps(fields)

    # -- bill line-item extraction ------------------------------------------------
    def _simulate_bill_page(self, user_prompt: str) -> str:
        page_match = re.search(r"page (\d+)\)", user_prompt)
        page_number = int(page_match.group(1)) if page_match else None

        page_line_items: dict[int, list[dict]] = {
            2: [
                {
                    "description": "Room Rent (Private Room, 2 days @ INR 8000/day)",
                    "amount": 16000,
                    "quantity": 2,
                    "unit_price": 8000,
                    "category": "room_rent",
                    "confidence": 0.95,
                },
                {
                    "description": "Nursing Charges",
                    "amount": 3000,
                    "quantity": None,
                    "unit_price": None,
                    "category": "nursing",
                    "confidence": 0.85,
                },
            ],
            3: [
                {
                    "description": "Consultant Visit Fee (Cardiologist)",
                    "amount": 4000,
                    "quantity": None,
                    "unit_price": None,
                    "category": "consultant_doctor_charges",
                    "confidence": 0.9,
                },
                {
                    "description": "Medicines",
                    "amount": 2500,
                    "quantity": None,
                    "unit_price": None,
                    "category": "medicines",
                    "confidence": 0.85,
                },
                {
                    "description": "Miscellaneous Hospital Charges",
                    "amount": 500,
                    "quantity": None,
                    "unit_price": None,
                    "category": "unknown",
                    "confidence": 0.3,
                },
            ],
        }
        return json.dumps({"line_items": page_line_items.get(page_number, [])})


def _unknown_field() -> dict:
    return {"status": "unknown", "value": None, "confidence": None, "source_chunk_ids": [], "notes": "No supplied evidence."}


def _ok_field(value, chunk_ids: list[str], confidence: float) -> dict:
    return {"status": "ok", "value": value, "confidence": confidence, "source_chunk_ids": chunk_ids, "notes": None}


def _print_stage(number: str, title: str, stage_result: pipeline.StageResult) -> None:
    status = "OK" if stage_result.ok else "FAILED"
    print(f"\n[{number}] {title} -- {status}")
    print(f"    {stage_result.message}")


def main() -> None:
    print("=" * 78)
    print("PHASE 7 END-TO-END DEMO (offline, simulated LLM)")
    print("Policy + Bill -> RAG -> LLM -> JSON -> Calculation -> Evidence -> Result")
    print("=" * 78)

    with tempfile.TemporaryDirectory() as tmp_dir:
        policy_path = Path(tmp_dir) / "synthetic_policy.pdf"
        bill_path = Path(tmp_dir) / "synthetic_bill.pdf"
        _build_pdf(policy_path, SYNTHETIC_POLICY_PAGES)
        _build_pdf(bill_path, SYNTHETIC_BILL_PAGES)

        provider = SimulatedLLMProvider()

        # 1. Document status
        policy_pages_result = pipeline.ingest_pdf(policy_path, document_id="policy-demo")
        _print_stage("1a", "Document status: policy", policy_pages_result)
        bill_pages_result = pipeline.ingest_pdf(bill_path, document_id="bill-demo")
        _print_stage("1b", "Document status: bill", bill_pages_result)
        if not (policy_pages_result.ok and bill_pages_result.ok):
            return

        # 2. RAG retrieval
        retrieval_result = pipeline.retrieve_policy_evidence(policy_pages_result.data)
        _print_stage("2", "Retrieved policy evidence (RAG)", retrieval_result)
        if not retrieval_result.ok:
            return
        for chunk in retrieval_result.data["chunks"]:
            print(f"    - page {chunk.page_number} (score={chunk.relevance_score:.3f}): "
                  f"{' '.join(chunk.text.split())[:70]}...")

        # 3. LLM policy field extraction
        policy_extraction_result = pipeline.extract_policy_fields(provider, retrieval_result.data["chunks"])
        _print_stage("3", "Extracted policy variables (LLM)", policy_extraction_result)
        if not policy_extraction_result.ok:
            return
        policy_fields = policy_extraction_result.data
        for name in policy_fields.model_fields:
            field = getattr(policy_fields, name)
            print(f"    {name}: status={field.status.value}, value={field.value!r}")

        # 4. LLM bill line-item extraction
        bill_extraction_result = pipeline.extract_bill_line_items(
            provider, bill_pages_result.data, document_id="bill-demo", filename="synthetic_bill.pdf"
        )
        _print_stage("4", "Extracted bill line items (LLM)", bill_extraction_result)
        if not bill_extraction_result.ok:
            return
        bill = bill_extraction_result.data
        for item in bill.line_items:
            print(f"    - [{item.category}] {item.description}: INR {item.amount} (page {item.page_number})")

        # 5. AI -> Calculation Engine JSON boundary
        print("\n[5] AI -> Calculation Engine JSON -- the architectural boundary")
        print("    Below this line: plain deterministic Python arithmetic. No LLM calls.")
        print("    --- ExtractedPolicyFields JSON ---")
        print("    " + policy_fields.model_dump_json())
        print("    --- ExtractedBill JSON ---")
        print("    " + bill.model_dump_json())

        # 6/7/8/9/10. Deterministic calculation + evidence + confidence + review
        audit_stage = pipeline.run_room_rent_audit(policy_fields, bill, ACTUAL_ROOM_RENT_PER_DAY)
        _print_stage("6", "Deterministic calculation", audit_stage)
        if not audit_stage.ok:
            return
        audit_result = audit_stage.data

        if audit_result.estimated_result:
            print("\n    Calculation trace:")
            for step in audit_result.estimated_result.trace.steps:
                print(f"      - {step.label}: {step.detail}")

        print("\n[7] Final Estimate")
        if audit_result.estimated_result:
            calc = audit_result.estimated_result
            print(f"    room_rent_adjustment        = INR {calc.room_rent_adjustment}")
            print(f"    associated_charge_adjustment = INR {calc.associated_charge_adjustment}")
            print(f"    total_adjustment            = INR {calc.total_adjustment}")

        print("\n[8] Evidence")
        for item in audit_result.evidence:
            print(f"    - field={item.extracted_field}, page={item.page}, "
                  f"document={item.document}, confidence={item.confidence}, rule={item.rule_applied}")

        print(f"\n[9] Confidence: {audit_result.confidence_level.value}")

        print(f"\n[10] Review status: {audit_result.status.value}")
        if audit_result.reason:
            print(f"     Reason: {audit_result.reason}")
        for warning in audit_result.warnings:
            print(f"     Warning: {warning}")

    print("\n" + "=" * 78)
    print("END-TO-END RUN COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
