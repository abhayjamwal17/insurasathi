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

## Phase 2 (this phase)

Phase 2 turns Phase 1's page-aware chunks into a working retrieval system —
retrieval only, no LLM, no answer generation, no calculation:

1. **Embeddings** (`rag/embeddings.py`) — `EmbeddingModel` wraps a local
   `sentence-transformers` model (`all-MiniLM-L6-v2` by default, ~80MB,
   CPU-only, no external API calls). Vectors are L2-normalized.
2. **Vector store** (`rag/vector_store.py`) — `ChunkVectorStore` indexes
   `Chunk` objects into a Chroma collection, keeping `document_id`,
   `filename`, `page_number`, and `chunk_id` as metadata on every vector.
   Cosine similarity space.
3. **Retriever** (`rag/retriever.py`) — `Retriever.retrieve(query)` takes a
   natural-language query and returns a `RetrievalResult` with the matched
   chunks (text, page number, filename, chunk ID, relevance score) sorted
   by relevance. `top_k` and `relevance_threshold` are both configurable.
   If nothing clears the threshold (or the store is empty), it returns
   `status="insufficient_evidence"` with no chunks — it never forces an
   answer.
4. **Schemas** (`rag/schemas.py`) — `RetrievedChunk` and `RetrievalResult`
   Pydantic models.
5. **Tests** (`tests/test_retriever.py`) — relevant queries retrieve the
   right clause, page/document metadata survives retrieval, multiple
   relevant chunks can come back, an irrelevant query yields
   `insufficient_evidence`, and an empty vector store fails gracefully.
   Uses a clearly-labeled synthetic test policy (`tests/conftest.py`), not
   real insurance content.
6. **Demo script** (`rag_demo.py`) — builds the same synthetic policy,
   indexes it, and runs several queries end to end. Run with
   `python rag_demo.py`.

## Phase 3 (this phase)

Phase 3 adds an LLM interpretation layer on top of Phase 2's retrieval.
**The LLM never calculates anything** — it interprets retrieved evidence
into structured fields; arithmetic is deliberately left for a later,
deterministic phase.

1. **Provider abstraction** (`extraction/providers.py`) — `LLMProvider` is
   an abstract interface with `GroqProvider` and `OllamaProvider`
   implementations, switchable via `LLM_PROVIDER=groq|ollama` in `.env`
   (see `get_provider()`). Neither provider is imported/required unless
   used — no API key is hardcoded anywhere. A third implementation,
   `StaticProvider`, returns a fixed canned response and exists purely for
   deterministic tests/offline demos (no network, no LLM needed).
2. **Structured schema** (`extraction/schemas.py`) — `ExtractedPolicyFields`
   holds one `PolicyField` per tracked fact (`room_rent_cap`,
   `room_rent_cap_unit`, `proportionate_deduction_applicable`,
   `copay_percentage`, `deductible`, `relevant_sub_limits`,
   `potentially_relevant_exclusions`). Every `PolicyField` carries
   `status` (`ok` / `unknown` / `needs_review`), `value`, `confidence`,
   `notes`, and a `sources: list[SourceEvidence]` — never a bare value.
3. **Prompt** (`extraction/prompts.py`) — `EXTRACTION_SYSTEM_PROMPT`
   explicitly instructs the model to use only the supplied evidence, never
   fall back on general insurance knowledge, never invent a value, cite
   the `chunk_id`(s) it used, return `unknown` when evidence is absent, and
   return `needs_review` (not a guess) when two passages conflict.
4. **Extractor** (`extraction/extractor.py`) — `PolicyExtractor.extract()`
   runs Evidence -> LLM -> structured field. Crucially, it never trusts the
   LLM's own claims about page numbers: it only trusts which `chunk_id`s
   the LLM cited, then looks those chunk_ids up in the evidence that was
   actually supplied and builds `SourceEvidence` from that — so a
   hallucinated page number in the model's prose can't leak into
   provenance. An unrecognized `chunk_id` is silently dropped, not trusted.
   Empty evidence short-circuits to all-`unknown` without even calling the
   LLM.
5. **Tests** (`tests/test_extraction.py`) — a clear room-rent clause, a
   clear co-pay clause, missing information (-> `unknown`, not a guess),
   conflicting information (-> `needs_review`, not a choice), page
   metadata surviving extraction (and being immune to what the LLM claims
   inline), unrecognized chunk_ids being dropped, empty evidence
   short-circuiting, and malformed/incomplete LLM output raising
   `ExtractionError`. All run offline via `StaticProvider` — no arithmetic
   is evaluated in this phase.
6. **Demo script** (`extraction_demo.py`) — runs full
   retrieval -> extraction on the same synthetic policy, printing extracted
   fields with page citations, and a second example showing the extractor
   correctly return `unknown` for a fact the evidence never states. Uses a
   `SimulatedLLMProvider` by default (this sandbox has no `GROQ_API_KEY`
   and no local Ollama server); pass `--live` with `LLM_PROVIDER` and
   credentials set in `.env` to run it against a real Groq or Ollama model
   instead — no code changes needed, just the provider swap.

Not implemented yet: the deterministic calculation engine, evidence/
confidence assembly beyond what extraction already carries, or any UI
(Streamlit etc.) — see "What's next" below.

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
repo. The first run downloads the `all-MiniLM-L6-v2` embedding model
(~80MB) from Hugging Face and caches it locally.

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
to exactly one page, which is what lets retrieval answer "what page did
this come from?"

## Data structures produced by retrieval

```python
class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    relevance_score: float   # cosine similarity to the query
    section: str | None

class RetrievalResult(BaseModel):
    query: str
    status: Literal["ok", "insufficient_evidence"]
    chunks: list[RetrievedChunk]
```

## Data structures produced by extraction

```python
class SourceEvidence(BaseModel):
    document_id: str
    filename: str
    page_number: int
    chunk_id: str
    text: str

class PolicyField(BaseModel):
    status: Literal["ok", "unknown", "needs_review"]
    value: float | int | str | bool | list[str] | None
    confidence: float | None       # 0.0-1.0, only set when status == "ok"
    sources: list[SourceEvidence]  # empty unless status == "ok"/"needs_review"
    notes: str | None

class ExtractedPolicyFields(BaseModel):
    room_rent_cap: PolicyField
    room_rent_cap_unit: PolicyField
    proportionate_deduction_applicable: PolicyField
    copay_percentage: PolicyField
    deductible: PolicyField
    relevant_sub_limits: PolicyField
    potentially_relevant_exclusions: PolicyField
```

## What remains for Phase 4+

- Deterministic calculation engine that applies `ExtractedPolicyFields` to
  a parsed hospital bill to produce estimated coverage / out-of-pocket
  amounts (`calculation/`) — plain Python arithmetic, no LLM involvement
- Evidence assembly + confidence scoring at the claim level, tying every
  calculated number back to the source chunk(s)/page(s)/field(s) it came
  from (`evidence/`)
- Human review / abstention flow for low-confidence or `needs_review`
  results
- Application UI (a later concern — separate from this technical PoC)
