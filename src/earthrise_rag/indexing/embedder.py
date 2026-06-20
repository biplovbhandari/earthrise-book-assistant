from __future__ import annotations


class LocalEmbeddingModel:
    def __init__(self, model_name: str, cache_dir: str = "/models") -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name, cache_folder=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def get_dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()
