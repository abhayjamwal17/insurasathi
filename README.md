# ICU / Hospital Interim Bill Auditor

A technical proof-of-concept for an MBA capstone project. The end goal is a
pipeline that reads an insurance policy PDF and an interim hospital bill,
and estimates coverage / out-of-pocket impact — with every number traceable
to a specific page of the source document, and **all arithmetic performed
by deterministic Python code, never by an LLM.**

Full intended architecture:

```
Insurance Policy PDF
        |
Page-Aware PDF Extraction
        |
Chunking + Metadata
        |
Vector Retrieval / RAG
        |
LLM Document Interpretation
        |
Structured JSON
        |
Pydantic Validation
        |
Deterministic Calculation Engine
        |
Estimated Coverage / Out-of-Pocket Impact
        |
Evidence + Confidence
        |
Human Review / Abstention
```

The LLM's job stops at "structured JSON." Everything from validation
onward is ordinary, auditable Python — important for a capstone viva where
every number needs to be explainable.

## Phase 1 (this phase)

Phase 1 implements only the ingestion foundation the rest of the pipeline
will sit on:

1. **Project structure** — `ingestion/`, `rag/`, `extraction/`,
   `calculation/`, `evidence/` are stubbed out as empty packages so the
   intended architecture is visible, but only `ingestion/` has real code.
2. **Configuration** (`config.py`) — reads settings from environment
   variables via `.env` (see `.env.example`). No secrets or API keys are
   hardcoded anywhere.
3. **Page-aware PDF ingestion** (`ingestion/pdf_parser.py`) — extracts text
   from a PDF one page at a time using PyMuPDF, preserving
   `document_id` / `filename` / `page_number` for every page. The document
   is never flattened into one blob.
4. **Chunking** (`ingestion/chunker.py`) — deterministic, character-based,
   configurable chunk size/overlap. A chunk never spans more than one
   source page, so every chunk keeps its page provenance. Section headings
   are only attached when a simple heuristic reliably detects one; nothing
   is guessed.
5. **Schemas** (`ingestion/schemas.py`) — `DocumentPage` and `Chunk`
   Pydantic models.
6. **Tests** (`tests/`) — cover PDF loading, page count, text extraction,
   page/metadata provenance, chunk provenance, and graceful failure on
   missing/empty/invalid PDFs, using a synthetic PDF built at test time
   (not real policy content).

Not implemented yet: retrieval/embeddings, LLM extraction, calculation,
evidence assembly, or any UI (Streamlit etc.) — see "What's next" below.

## Install

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
cp .env.example .env          # optional — defaults work without it
```

## Run the ingestion demo

```bash
python app.py path\to\some.pdf
```

This prints the page count, the number of chunks produced, and a preview
of the first few chunks with their page numbers.

## Run the tests

```bash
pytest -v
```

All tests use a synthetic PDF generated on the fly in
`tests/conftest.py` — no real insurance documents are checked into this
repo.

## Data structures produced by the parser

```python
class DocumentPage(BaseModel):
    document_id: str
    filename: str
    page_number: int   # 1-indexed
    text: str

class Chunk(BaseModel):
    document_id: str
    filename: str
    page_number: int   # matches exactly one DocumentPage
    chunk_id: str
    text: str
    section: str | None   # None unless reliably detected
```

`parse_pdf(path)` returns `list[DocumentPage]` — one entry per page, in
order. `chunk_pages(pages)` returns `list[Chunk]` — every chunk maps back
to exactly one page, which is what lets a later retrieval step answer
"what page did this come from?"

## What remains for Phase 2+

- Vector embeddings + retrieval over `Chunk` objects (`rag/`)
- LLM-based structured extraction of policy terms / bill line items into
  Pydantic models (`extraction/`) — interpretation only, no arithmetic
- Deterministic calculation engine that applies coverage rules to produce
  estimated coverage / out-of-pocket amounts (`calculation/`)
- Evidence assembly + confidence scoring, tying every calculated number
  back to the source chunk(s)/page(s) it came from (`evidence/`)
- Human review / abstention flow for low-confidence results
- Application UI (a later concern — separate from this technical PoC)
