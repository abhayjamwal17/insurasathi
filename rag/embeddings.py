"""Local sentence-embedding wrapper.

Uses sentence-transformers with a small model (default: all-MiniLM-L6-v2,
~80MB) that runs on CPU with no external API calls. Embeddings are
L2-normalized so downstream cosine similarity is a plain dot product.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """Thin wrapper around a local sentence-transformers model."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one normalized vector per text."""
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([text])[0]
