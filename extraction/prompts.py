"""Prompt construction for LLM policy-field extraction.

The LLM is shown only the retrieved evidence chunks (never the whole
policy) and is instructed, explicitly and repeatedly, to interpret that
evidence rather than fall back on outside insurance knowledge.
"""

from __future__ import annotations

import json

from extraction.schemas import FIELD_DESCRIPTIONS, FIELD_NAMES
from rag.schemas import RetrievedChunk

EXTRACTION_SYSTEM_PROMPT = """\
You are a policy-interpretation assistant for an insurance bill auditor.

You will be given a list of evidence passages retrieved from an insurance \
policy document, each tagged with a chunk_id and a page number. You must \
extract a fixed set of structured fields from that evidence.

Rules you MUST follow exactly:

1. Use ONLY the evidence passages supplied in the user message. Do not use \
outside or general insurance knowledge to fill in missing information, \
even if you believe you know the typical value.
2. Do not invent, estimate, or guess any value that is not explicitly \
stated in the supplied evidence.
3. For every field, cite the chunk_id(s) of the evidence you used to reach \
that field's value, in "source_chunk_ids". Only use chunk_ids that were \
given to you — never invent one.
4. If the supplied evidence does not establish a value for a field, set \
"status" to "unknown" and set "value" to null for that field.
5. If two or more supplied evidence passages state conflicting values for \
the same field, do NOT pick one over the other. Set "status" to \
"needs_review", set "value" to null, cite all conflicting chunk_ids in \
"source_chunk_ids", and explain the conflict in "notes".
6. When status is "ok", set "confidence" to a number between 0 and 1 \
reflecting how directly the evidence states the value. When status is \
"unknown" or "needs_review", set "confidence" to null.
7. Respond with a single JSON object only — no markdown, no commentary \
before or after it — matching exactly the schema you are given.
"""


def _format_evidence_block(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for chunk in chunks:
        lines.append(
            f"[chunk_id={chunk.chunk_id} | page={chunk.page_number} | "
            f"document={chunk.filename}]\n{chunk.text}"
        )
    return "\n\n".join(lines)


def _format_schema_block() -> str:
    fields = {
        name: {
            "description": FIELD_DESCRIPTIONS[name],
            "status": "ok | unknown | needs_review",
            "value": "<value, or null>",
            "confidence": "<0.0-1.0, or null>",
            "source_chunk_ids": ["<chunk_id, ...>"],
            "notes": "<string, or null>",
        }
        for name in FIELD_NAMES
    }
    return json.dumps(fields, indent=2)


def build_extraction_prompt(chunks: list[RetrievedChunk]) -> str:
    """Build the user-turn prompt: evidence + required output schema."""
    evidence_block = _format_evidence_block(chunks)
    schema_block = _format_schema_block()

    return f"""\
EVIDENCE (retrieved policy passages — this is all the information you may use):

{evidence_block}

Extract the following fields from the evidence above. Return a single JSON \
object with exactly these top-level keys, each holding an object shaped \
like the template below (the "description" text is guidance for you, not \
part of the output shape — omit it from your JSON):

{schema_block}
"""
