"""Query-facing retrieval API.

Wraps a ChunkVectorStore with a configurable top_k and relevance threshold.
If no candidate clears the threshold (or the store is empty), retrieve()
returns an explicit "insufficient_evidence" result instead of forcing a
best-effort answer — the calling code (later: LLM interpretation) must
handle that state rather than assume evidence always exists.
"""

from __future__ import annotations

from rag.schemas import RetrievalResult, RetrievedChunk
from rag.vector_store import ChunkVectorStore

DEFAULT_TOP_K = 5
DEFAULT_RELEVANCE_THRESHOLD = 0.35


class Retriever:
    """Retrieves relevant, page-attributed chunks for a natural-language query."""

    def __init__(
        self,
        vector_store: ChunkVectorStore,
        top_k: int = DEFAULT_TOP_K,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ) -> None:
        self._store = vector_store
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        relevance_threshold: float | None = None,
    ) -> RetrievalResult:
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        effective_top_k = top_k if top_k is not None else self.top_k
        effective_threshold = (
            relevance_threshold if relevance_threshold is not None else self.relevance_threshold
        )

        if self._store.count() == 0:
            return RetrievalResult(query=query, status="insufficient_evidence", chunks=[])

        candidates: list[RetrievedChunk] = self._store.query(query, top_k=effective_top_k)
        relevant = [c for c in candidates if c.relevance_score >= effective_threshold]

        if not relevant:
            return RetrievalResult(query=query, status="insufficient_evidence", chunks=[])

        relevant.sort(key=lambda c: c.relevance_score, reverse=True)
        return RetrievalResult(query=query, status="ok", chunks=relevant)
