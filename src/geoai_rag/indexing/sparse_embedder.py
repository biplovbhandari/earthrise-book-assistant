from __future__ import annotations

_IDF_MODELS = {"Qdrant/bm25", "Qdrant/bm42-all-minilm-l6-v2-attentions"}


class LocalSparseEmbeddingModel:
    """Sparse embedding adapter using fastembed models loaded locally."""

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        """Load a fastembed sparse model into memory."""
        from fastembed import SparseTextEmbedding

        self._model_name = model_name
        self._model = SparseTextEmbedding(model_name=model_name, cache_dir=cache_dir)

    @property
    def requires_idf(self) -> bool:
        """BM25 and BM42 need Qdrant's IDF modifier; SPLADE does not."""
        return self._model_name in _IDF_MODELS

    def embed(self, texts: list[str]) -> list:
        """Encode a batch of document texts into sparse vectors."""
        return list(self._model.embed(texts))

    def query_embed(self, text: str) -> list:
        """Encode a single query string into a sparse vector."""
        return list(self._model.query_embed(text))
