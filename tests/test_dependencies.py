from unittest.mock import MagicMock

from earthrise_rag.config import Settings


class FakeEmbedder:
    def get_dimension(self):
        return 10


class FakeStore:
    def __init__(self, *args, **kwargs):
        self.create_if_missing = kwargs.get("create_if_missing", True)
        self._has_sparse = True
        self.sparse_embedder: object | None = None

    def count(self):
        return 100

    def search_sparse(self, text, top_k=10, filters=None):
        return []


class FakeLLMClient:
    def chat(self, messages, temperature=0.3, max_tokens=1024):
        return "fake"


class FakeSparseEmbedder:
    pass


def _patch_adapters(monkeypatch):
    monkeypatch.setattr(
        "api.dependencies._create_embedder",
        lambda config: FakeEmbedder(),
    )

    def fake_store_factory(config, dim, create_if_missing, sparse_embedder=None):
        store = FakeStore(create_if_missing=create_if_missing)
        store.sparse_embedder = sparse_embedder
        return store

    monkeypatch.setattr(
        "api.dependencies._create_vector_store",
        fake_store_factory,
    )
    monkeypatch.setattr(
        "api.dependencies._create_llm_client",
        lambda config: FakeLLMClient(),
    )
    monkeypatch.setattr(
        "api.dependencies._create_sparse_embedder",
        lambda config: FakeSparseEmbedder(),
    )


def test_create_pipelines_wires_dense_strategy(monkeypatch):
    _patch_adapters(monkeypatch)
    settings = Settings(
        retrieval_strategy="dense",
        qdrant_url="http://fake:6333",
    )

    from api.dependencies import create_pipelines
    from earthrise_rag.citations import DefaultCitationBuilder
    from earthrise_rag.generation import DefaultContextBuilder
    from earthrise_rag.query import QueryPipeline
    from earthrise_rag.retrieval import DenseStrategy, NoOpReranker

    pipelines = create_pipelines(settings)

    assert pipelines.query is not None
    assert isinstance(pipelines.query, QueryPipeline)
    assert isinstance(pipelines.query._strategy, DenseStrategy)
    assert isinstance(pipelines.query._strategy._reranker, NoOpReranker)
    assert pipelines.vector_store is not None
    assert pipelines.query._context_builder is not None
    assert isinstance(pipelines.query._context_builder, DefaultContextBuilder)
    assert pipelines.query._llm_client is not None
    assert isinstance(pipelines.query._llm_client, FakeLLMClient)
    assert pipelines.query._citation_builder is not None
    assert isinstance(pipelines.query._citation_builder, DefaultCitationBuilder)
    assert getattr(pipelines.vector_store, "sparse_embedder", None) is None


def test_create_pipelines_isolates_llm_failure(monkeypatch):
    """LLM creation failure must not prevent retrieval from working."""
    _patch_adapters(monkeypatch)
    monkeypatch.setattr(
        "api.dependencies._create_llm_client",
        MagicMock(side_effect=Exception("openai not installed")),
    )
    settings = Settings(
        retrieval_strategy="dense",
        qdrant_url="http://fake:6333",
    )

    from api.dependencies import create_pipelines
    from earthrise_rag.query import QueryPipeline

    pipelines = create_pipelines(settings)

    assert pipelines.query is not None
    assert isinstance(pipelines.query, QueryPipeline)
    assert pipelines.query._llm_client is None
    assert pipelines.vector_store is not None


def test_create_pipelines_wires_hybrid_strategy(monkeypatch):
    _patch_adapters(monkeypatch)
    settings = Settings(
        retrieval_strategy="hybrid",
        qdrant_url="http://fake:6333",
    )

    from api.dependencies import create_pipelines
    from earthrise_rag.retrieval import HybridStrategy, NoOpReranker

    pipelines = create_pipelines(settings)

    assert pipelines.query is not None
    assert isinstance(pipelines.query._strategy, HybridStrategy)
    assert isinstance(pipelines.query._strategy._reranker, NoOpReranker)
    assert pipelines.query._strategy._rrf_k == 60
    assert getattr(pipelines.vector_store, "sparse_embedder", None) is not None


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


def test_create_indexing_pipeline_registers_pdf_and_json(monkeypatch):
    _patch_adapters(monkeypatch)
    monkeypatch.setattr(
        "api.dependencies._load_video_chapter_map",
        lambda: {},
    )
    settings = Settings(
        retrieval_strategy="hybrid",
        qdrant_url="http://fake:6333",
    )

    from api.dependencies import create_indexing_pipeline

    pipeline = create_indexing_pipeline(settings)

    assert ".pdf" in pipeline.parsers
    assert ".json" in pipeline.parsers
    assert ".pdf" in pipeline.chunkers
    assert ".json" in pipeline.chunkers
