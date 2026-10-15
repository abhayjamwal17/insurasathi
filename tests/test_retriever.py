"""Tests for the Phase 2 RAG retrieval layer.

Uses the synthetic, clearly-labeled test policy from conftest.py — not real
insurance content.
"""

from __future__ import annotations

from rag.retriever import Retriever
from rag.vector_store import ChunkVectorStore, VectorStoreError

import pytest


def test_relevant_query_retrieves_relevant_clause(policy_vector_store: ChunkVectorStore) -> None:
    retriever = Retriever(policy_vector_store, top_k=3, relevance_threshold=0.2)
    result = retriever.retrieve("What is the room rent limit?")

    assert result.status == "ok"
    assert len(result.chunks) > 0
    top = result.chunks[0]
    assert "room rent" in top.text.lower()
    assert "5000" in top.text or "one percent" in top.text.lower()


def test_page_metadata_survives_retrieval(policy_vector_store: ChunkVectorStore) -> None:
    retriever = Retriever(policy_vector_store, top_k=3, relevance_threshold=0.2)
    result = retriever.retrieve("What is the room rent limit?")

    top = result.chunks[0]
    # ROOM RENT LIMIT is page 2 of the synthetic policy fixture.
    assert top.page_number == 2


def test_document_metadata_survives_retrieval(policy_vector_store: ChunkVectorStore) -> None:
    retriever = Retriever(policy_vector_store, top_k=3, relevance_threshold=0.2)
    result = retriever.retrieve("What is the room rent limit?")

    top = result.chunks[0]
    assert top.filename == "synthetic_policy.pdf"
    assert top.document_id == "synthetic-policy-001"
    assert top.chunk_id


def test_multiple_relevant_chunks_can_be_returned(policy_vector_store: ChunkVectorStore) -> None:
    retriever = Retriever(policy_vector_store, top_k=6, relevance_threshold=0.0)
    result = retriever.retrieve("hospitalization expenses and charges")

    assert result.status == "ok"
    assert len(result.chunks) > 1


def test_irrelevant_query_produces_insufficient_evidence(
    policy_vector_store: ChunkVectorStore,
) -> None:
    retriever = Retriever(policy_vector_store, top_k=3, relevance_threshold=0.6)
    result = retriever.retrieve("What is the capital city of France?")

    assert result.status == "insufficient_evidence"
    assert result.chunks == []


def test_empty_vector_store_fails_gracefully(empty_vector_store: ChunkVectorStore) -> None:
    retriever = Retriever(empty_vector_store, top_k=3, relevance_threshold=0.2)
    result = retriever.retrieve("What is the room rent limit?")

    assert result.status == "insufficient_evidence"
    assert result.chunks == []


def test_empty_query_raises_value_error(policy_vector_store: ChunkVectorStore) -> None:
    retriever = Retriever(policy_vector_store)
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_vector_store_query_on_empty_store_returns_empty_list(
    empty_vector_store: ChunkVectorStore,
) -> None:
    assert empty_vector_store.query("anything", top_k=3) == []


def test_vector_store_empty_query_text_raises(policy_vector_store: ChunkVectorStore) -> None:
    with pytest.raises(VectorStoreError):
        policy_vector_store.query("", top_k=3)


def test_retrieved_chunk_relevance_scores_are_descending(
    policy_vector_store: ChunkVectorStore,
) -> None:
    retriever = Retriever(policy_vector_store, top_k=6, relevance_threshold=0.0)
    result = retriever.retrieve("intensive care unit charges")

    scores = [c.relevance_score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)
