from __future__ import annotations

import math

from earthrise_rag.models.scored_chunk import ScoredChunk

_NON_FINITE_SENTINEL = -1e9


class LocalCrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers models."""

    def __init__(self, model_name: str, cache_dir: str = "/models") -> None:
        """Load a cross-encoder model and verify it is a single-label scorer."""
        from sentence_transformers import CrossEncoder

        self._model_name = model_name
        self._model = CrossEncoder(model_name, cache_folder=cache_dir)

        if self._model.num_labels != 1:
            raise ValueError(
                f"CrossEncoder '{model_name}' has {self._model.num_labels} labels, "
                f"expected 1 (single-label relevance scorer). "
                f"Check RERANKER_MODEL_NAME."
            )

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        """Re-score candidates using cross-encoder and return top_k by relevance."""
        import numpy as np

        if not candidates:
            return []

        pairs = [(query, sc.chunk.content) for sc in candidates]
        raw_scores = self._model.predict(pairs, show_progress_bar=False)

        scores = np.asarray(raw_scores)
        if scores.ndim != 1 or len(scores) != len(candidates):
            raise ValueError(
                f"CrossEncoder returned shape {scores.shape}, "
                f"expected ({len(candidates)},). Check RERANKER_MODEL_NAME is a single-label model."
            )

        scored = []
        for sc, s in zip(candidates, scores):
            score = float(s)
            if not math.isfinite(score):
                score = _NON_FINITE_SENTINEL
            scored.append(ScoredChunk(chunk=sc.chunk, score=score, ranking_method="reranked"))

        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:top_k]


class NoOpReranker:
    """Passthrough reranker that preserves the original ranking order."""

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        """Return the first top_k candidates unchanged."""
        return candidates[:top_k]
