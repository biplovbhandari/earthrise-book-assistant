import pytest

from earthrise_rag.models import Chunk, ScoredChunk
from earthrise_rag.retrieval import DenseStrategy, HybridStrategy, NoOpReranker


def _make_scored_chunk(content: str, score: float, method: str = "dense") -> ScoredChunk:
    chunk = Chunk(
        content=content,
        content_hash="abc",
        source_type="book_text",
        content_type="concept",
    )
    return ScoredChunk(chunk=chunk, score=score, ranking_method=method)


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.1] * 10 for _ in texts]

    def embed_query(self, text):
        return [0.5] * 10

    def get_dimension(self):
        return 10


class FakeStore:
    def __init__(self, results, sparse_results=None):
        self._results = results
        self._sparse_results = sparse_results or []

    def upsert(self, chunks, vectors):
        pass

    def search_dense(self, vector, top_k=10, filters=None):
        return self._results[:top_k]

    def search_sparse(self, text, top_k=10, filters=None):
        return self._sparse_results[:top_k]

    def get_by_ids(self, ids):
        return []

    def delete_by_source(self, source_path):
        pass

    def count(self):
        return 100


# --- DenseStrategy tests ---


def test_dense_strategy_retrieves_and_reranks():
    canned = [
        _make_scored_chunk("U-Net architecture", 0.95),
        _make_scored_chunk("CNN basics", 0.80),
    ]
    store = FakeStore(canned)
    strategy = DenseStrategy(FakeEmbedder(), store, NoOpReranker())

    results = strategy.retrieve("What is U-Net?", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.content == "U-Net architecture"
    assert results[0].score == 0.95
    assert results[1].chunk.content == "CNN basics"


# --- HybridStrategy tests ---


def test_hybrid_strategy_fuses_dense_and_sparse():
    shared_chunk = _make_scored_chunk("U-Net architecture", 0.95)
    dense_only = _make_scored_chunk("CNN basics", 0.80)
    sparse_only = _make_scored_chunk("NDVI index", 0.70, method="sparse")
    sparse_shared = _make_scored_chunk("U-Net architecture", 0.60, method="sparse")
    sparse_shared.chunk = shared_chunk.chunk

    store = FakeStore(
        results=[shared_chunk, dense_only],
        sparse_results=[sparse_shared, sparse_only],
    )
    strategy = HybridStrategy(FakeEmbedder(), store, NoOpReranker(), rrf_k=60)

    results = strategy.retrieve("What is U-Net?", top_k=10)

    assert results[0].chunk.content == "U-Net architecture"
    assert results[0].ranking_method == "hybrid_rrf"
    assert results[0].score == pytest.approx(1 / 61 + 1 / 61)
    assert results[0].score > results[1].score

    contents = {r.chunk.content for r in results}
    assert "CNN basics" in contents
    assert "NDVI index" in contents


def test_hybrid_strategy_disjoint_results():
    dense = [_make_scored_chunk("Dense result A", 0.9)]
    sparse = [_make_scored_chunk("Sparse result B", 0.8, method="sparse")]

    store = FakeStore(results=dense, sparse_results=sparse)
    strategy = HybridStrategy(FakeEmbedder(), store, NoOpReranker(), rrf_k=60)

    results = strategy.retrieve("query", top_k=10)

    assert len(results) == 2
    contents = {r.chunk.content for r in results}
    assert "Dense result A" in contents
    assert "Sparse result B" in contents
    assert all(r.ranking_method == "hybrid_rrf" for r in results)


def test_hybrid_strategy_empty_sparse_skips_rrf():
    dense = [
        _make_scored_chunk("Dense result", 0.95),
    ]
    store = FakeStore(results=dense, sparse_results=[])
    strategy = HybridStrategy(FakeEmbedder(), store, NoOpReranker(), rrf_k=60)

    results = strategy.retrieve("query", top_k=5)

    assert len(results) == 1
    assert results[0].score == 0.95
    assert results[0].ranking_method == "dense"


def test_hybrid_strategy_both_empty():
    store = FakeStore(results=[], sparse_results=[])
    strategy = HybridStrategy(FakeEmbedder(), store, NoOpReranker(), rrf_k=60)

    results = strategy.retrieve("query", top_k=5)

    assert results == []


def test_hybrid_strategy_respects_top_k():
    dense = [_make_scored_chunk(f"Dense {i}", 0.9 - i * 0.1) for i in range(5)]
    sparse = [_make_scored_chunk(f"Sparse {i}", 0.8 - i * 0.1, method="sparse") for i in range(5)]

    store = FakeStore(results=dense, sparse_results=sparse)
    strategy = HybridStrategy(FakeEmbedder(), store, NoOpReranker(), rrf_k=60)

    results = strategy.retrieve("query", top_k=3)

    assert len(results) == 3
