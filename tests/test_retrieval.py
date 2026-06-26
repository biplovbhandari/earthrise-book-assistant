from earthrise_rag.models import Chunk, ScoredChunk
from earthrise_rag.retrieval import DenseStrategy, NoOpReranker


def _make_scored_chunk(content: str, score: float) -> ScoredChunk:
    chunk = Chunk(
        content=content,
        content_hash="abc",
        source_type="book_text",
        content_type="concept",
    )
    return ScoredChunk(chunk=chunk, score=score, ranking_method="dense")


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.1] * 10 for _ in texts]

    def embed_query(self, text):
        return [0.5] * 10

    def get_dimension(self):
        return 10


class FakeStore:
    def __init__(self, results):
        self._results = results

    def upsert(self, chunks, vectors):
        pass

    def search_dense(self, vector, top_k=10, filters=None):
        return self._results[:top_k]

    def search_sparse(self, text, top_k=10, filters=None):
        raise NotImplementedError

    def get_by_ids(self, ids):
        return []

    def delete_by_source(self, source_path):
        pass

    def count(self):
        return 100


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
