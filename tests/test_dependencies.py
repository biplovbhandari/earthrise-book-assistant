from earthrise_rag.config import Settings


class FakeEmbedder:
    def get_dimension(self):
        return 10


class FakeStore:
    def __init__(self, *args, **kwargs):
        self.create_if_missing = kwargs.get("create_if_missing", True)

    def count(self):
        return 100


def _patch_adapters(monkeypatch):
    monkeypatch.setattr(
        "api.dependencies._create_embedder",
        lambda config: FakeEmbedder(),
    )

    def fake_store_factory(config, dim, create_if_missing):
        return FakeStore(create_if_missing=create_if_missing)

    monkeypatch.setattr(
        "api.dependencies._create_vector_store",
        fake_store_factory,
    )


def test_create_pipelines_wires_dense_strategy(monkeypatch):
    _patch_adapters(monkeypatch)
    settings = Settings(
        retrieval_strategy="dense",
        qdrant_url="http://fake:6333",
    )

    from api.dependencies import create_pipelines
    from earthrise_rag.query import QueryPipeline
    from earthrise_rag.retrieval import DenseStrategy, NoOpReranker

    pipelines = create_pipelines(settings)

    assert pipelines.query is not None
    assert isinstance(pipelines.query, QueryPipeline)
    assert isinstance(pipelines.query._strategy, DenseStrategy)
    assert isinstance(pipelines.query._strategy._reranker, NoOpReranker)
    assert pipelines.vector_store is not None


def test_create_indexing_pipeline_no_strategy(monkeypatch):
    _patch_adapters(monkeypatch)
    settings = Settings(
        retrieval_strategy="hybrid",
        qdrant_url="http://fake:6333",
    )

    from api.dependencies import create_indexing_pipeline

    pipeline = create_indexing_pipeline(settings)

    assert pipeline is not None
    assert hasattr(pipeline, "index_source")
