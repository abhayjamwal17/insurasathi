"""Prompt construction for hospital bill line-item extraction.

The LLM is shown one page of bill text at a time and asked only to
identify and categorize line items — never to judge coverage, and never to
invent an amount or category the text doesn't support.
"""

from __future__ import annotations

from ingestion.schemas import DocumentPage
from extraction.bill_schemas import BILL_LINE_ITEM_CATEGORIES

BILL_EXTRACTION_SYSTEM_PROMPT = f"""\
You are a hospital-bill line-item extraction assistant for an insurance \
bill auditor.

You will be given the raw text of one page of a hospital interim bill. \
Hospitals use inconsistent terminology for the same kind of charge — you \
must interpret each line item's meaning, not just match keywords.

Rules you MUST follow exactly:

1. Extract every distinct billed line item on the page. Do not merge \
unrelated items and do not split one item into several.
2. Categorize each item using ONLY one of these controlled category \
values: {list(BILL_LINE_ITEM_CATEGORIES)}.
3. If you cannot confidently place an item into one of the specific \
categories, set "category" to "unknown". Do NOT guess a specific category \
just to avoid using "unknown".
4. If a line item's amount is not clearly stated as a number on the page, \
set "amount" to null. Do not invent, estimate, or infer an amount from \
context.
5. Set "quantity" and "unit_price" only when they are explicitly present \
for that item; otherwise null.
6. Set "confidence" to a number between 0 and 1 reflecting how clearly \
the description and category are supported by the text.
7. Do NOT decide whether any item is covered by an insurance policy, and \
do NOT perform any arithmetic (totals, subtotals, taxes). Extraction and \
categorization only.
8. If the page has no billable line items (e.g. a cover page, letterhead, \
or a page of free-text notes with no itemized charges), return an empty \
list.
9. Respond with a single JSON object only — no markdown, no commentary —\
 shaped exactly like: {{"line_items": [{{"description": "...", \
"amount": <number or null>, "quantity": <number or null>, \
"unit_price": <number or null>, "category": "...", \
"confidence": <0.0-1.0>}}, ...]}}
"""


def build_bill_extraction_prompt(page: DocumentPage) -> str:
    """Build the user-turn prompt for one bill page."""
    return f"""\
HOSPITAL BILL PAGE (document: {page.filename}, page {page.page_number}):

{page.text}

Extract the line items on this page as instructed. Return JSON only.
"""
