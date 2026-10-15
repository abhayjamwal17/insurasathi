"""Chroma-backed vector store for policy/bill chunks.

Every stored vector keeps its source Chunk's provenance
(document_id, filename, page_number, chunk_id) as Chroma metadata, so a
similarity match can always be traced back to an exact page.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from ingestion.schemas import Chunk
from rag.embeddings import DEFAULT_MODEL_NAME, EmbeddingModel
from rag.schemas import RetrievedChunk

DEFAULT_COLLECTION_NAME = "policy_chunks"


class VectorStoreError(Exception):
    """Raised for invalid vector store operations."""


class ChunkVectorStore:
    """Stores Chunk embeddings and supports similarity search over them."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: EmbeddingModel | None = None,
        persist_directory: str | Path | None = None,
    ) -> None:
        self._embedder = embedding_model or EmbeddingModel(DEFAULT_MODEL_NAME)

        if persist_directory is not None:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_directory))
        else:
            self._client = chromadb.EphemeralClient()

        # Cosine space matches the normalized embeddings from EmbeddingModel.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and index a batch of chunks. No-op for an empty list."""
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self._embedder.embed_texts(texts)

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "section": chunk.section or "",
                }
                for chunk in chunks
            ],
        )

    def count(self) -> int:
        return self._collection.count()

    def query(self, query_text: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the top_k most similar chunks to query_text, best first.

        Returns an empty list if the store is empty. Does not apply a
        relevance threshold — that is the Retriever's responsibility.
        """
        if not query_text or not query_text.strip():
            raise VectorStoreError("query_text must not be empty")

        available = self.count()
        if available == 0:
            return []

        query_embedding = self._embedder.embed_query(query_text)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, available),
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            # Chroma's cosine "distance" is (1 - cosine_similarity).
            similarity = 1.0 - distance
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=metadata["document_id"],
                    filename=metadata["filename"],
                    page_number=metadata["page_number"],
                    text=text,
                    relevance_score=similarity,
                    section=metadata.get("section") or None,
                )
            )

        return retrieved
