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


class FakeCrossEncoderReranker:
    def rerank(self, query, candidates, top_k):
        return candidates[:top_k]


def _patch_adapters(monkeypatch, reranker_factory=None):
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
    from earthrise_rag.retrieval.rerankers import NoOpReranker

    monkeypatch.setattr(
        "api.dependencies._create_reranker",
        reranker_factory if reranker_factory is not None else lambda config: NoOpReranker(),
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


# --- Cross-encoder reranker tests ---
def test_create_reranker_factory_wires_cross_encoder(monkeypatch):
    """Test _create_reranker directly with mocked CrossEncoder constructor."""
    from unittest.mock import MagicMock, patch

    mock_ce_class = MagicMock()
    mock_ce_instance = MagicMock()
    mock_ce_instance.num_labels = 1
    mock_ce_class.return_value = mock_ce_instance

    with patch("sentence_transformers.CrossEncoder", mock_ce_class):
        from api.dependencies import _create_reranker
        from earthrise_rag.retrieval.rerankers import LocalCrossEncoderReranker

        settings = Settings(
            reranker_provider="local_cross_encoder",
            reranker_model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
            qdrant_url="http://fake:6333",
        )

        reranker = _create_reranker(settings)

        assert isinstance(reranker, LocalCrossEncoderReranker)
        mock_ce_class.assert_called_once_with(
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            cache_folder=settings.hf_home,
        )


def test_create_pipelines_wires_cross_encoder_reranker(monkeypatch):
    _patch_adapters(monkeypatch, reranker_factory=lambda config: FakeCrossEncoderReranker())
    settings = Settings(
        retrieval_strategy="hybrid",
        reranker_provider="local_cross_encoder",
        qdrant_url="http://fake:6333",
    )

    from api.dependencies import create_pipelines
    from earthrise_rag.retrieval import HybridStrategy

    pipelines = create_pipelines(settings)

    assert pipelines.query is not None
    assert isinstance(pipelines.query._strategy, HybridStrategy)
    assert isinstance(pipelines.query._strategy._reranker, FakeCrossEncoderReranker)


def test_create_reranker_factory_rejects_empty_model_name():
    import pytest

    from api.dependencies import _create_reranker

    settings = Settings(
        reranker_provider="local_cross_encoder",
        reranker_model_name="",
        qdrant_url="http://fake:6333",
    )

    with pytest.raises(ValueError, match="RERANKER_MODEL_NAME must not be empty"):
        _create_reranker(settings)


def test_create_reranker_factory_tolerates_empty_model_when_noop():
    from api.dependencies import _create_reranker
    from earthrise_rag.retrieval import NoOpReranker

    settings = Settings(
        reranker_provider="noop",
        reranker_model_name="",
        qdrant_url="http://fake:6333",
    )

    reranker = _create_reranker(settings)
    assert isinstance(reranker, NoOpReranker)


def test_create_reranker_factory_rejects_multi_label_model():
    """Multi-label model should fail at startup, not at first query."""
    import pytest
    from unittest.mock import MagicMock, patch

    mock_ce_class = MagicMock()
    mock_ce_instance = MagicMock()
    mock_ce_instance.num_labels = 3
    mock_ce_class.return_value = mock_ce_instance

    with patch("sentence_transformers.CrossEncoder", mock_ce_class):
        from api.dependencies import _create_reranker

        settings = Settings(
            reranker_provider="local_cross_encoder",
            reranker_model_name="some-multi-label-model",
            qdrant_url="http://fake:6333",
        )

        with pytest.raises(ValueError, match="expected 1"):
            _create_reranker(settings)
