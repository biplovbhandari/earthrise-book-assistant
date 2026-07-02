from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import Distance


def _make_collection_mock(name):
    """Create a mock with .name set correctly (MagicMock(name=...) sets the label, not .name)."""
    m = MagicMock()
    m.name = name
    return m


def _mock_collection_with_sparse(name="test_collection", dense_dim=10):
    """Create a mock QdrantClient that returns a collection with both dense and sparse config."""
    mock_client = MagicMock()
    mock_collection_info = MagicMock()
    mock_vectors = {"dense": MagicMock(size=dense_dim, distance=Distance.COSINE)}
    mock_collection_info.config.params.vectors = mock_vectors
    mock_collection_info.config.params.sparse_vectors = {"sparse": MagicMock()}
    mock_client.get_collections.return_value.collections = [_make_collection_mock(name)]
    mock_client.get_collection.return_value = mock_collection_info
    return mock_client


def _mock_collection_dense_only(name="test_collection", dense_dim=10):
    """Create a mock QdrantClient that returns a collection with dense config only (no sparse)."""
    mock_client = MagicMock()
    mock_collection_info = MagicMock()
    mock_vectors = {"dense": MagicMock(size=dense_dim, distance=Distance.COSINE)}
    mock_collection_info.config.params.vectors = mock_vectors
    mock_collection_info.config.params.sparse_vectors = None
    mock_client.get_collections.return_value.collections = [_make_collection_mock(name)]
    mock_client.get_collection.return_value = mock_collection_info
    return mock_client


class FakeSparseEmbedding:
    indices = [1, 2, 3]
    values = [0.1, 0.2, 0.3]


class FakeSparseEmbedder:
    requires_idf = False

    def embed(self, texts):
        return [FakeSparseEmbedding() for _ in texts]

    def query_embed(self, text):
        return [FakeSparseEmbedding()]


def test_filter_key_mapping():
    with patch("qdrant_client.QdrantClient") as MockClient:
        mock_client = _mock_collection_with_sparse()
        MockClient.return_value = mock_client

        from earthrise_rag.indexing.qdrant_store import QdrantStore

        store = QdrantStore("http://fake:6333", "test_collection", 10)

        mock_client.query_points.return_value.points = []

        store.search_dense(
            [0.1] * 10, top_k=5, filters={"chapter": "03", "source_type": "book_text"}
        )

        call_args = mock_client.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter") or call_args[1].get("query_filter")

        conditions = query_filter.must
        keys = {c.key for c in conditions}
        assert "metadata.chapter" in keys
        assert "source_type" in keys


def test_search_sparse_builds_filter():
    with patch("qdrant_client.QdrantClient") as MockClient:
        mock_client = _mock_collection_with_sparse()
        MockClient.return_value = mock_client

        from earthrise_rag.indexing.qdrant_store import QdrantStore

        store = QdrantStore(
            "http://fake:6333", "test_collection", 10, sparse_embedder=FakeSparseEmbedder()
        )

        mock_client.query_points.return_value.points = []

        store.search_sparse("query text", top_k=5, filters={"chapter": "03"})

        call_args = mock_client.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter") or call_args[1].get("query_filter")

        conditions = query_filter.must
        keys = {c.key for c in conditions}
        assert "metadata.chapter" in keys


def test_upsert_includes_sparse_vectors():
    with patch("qdrant_client.QdrantClient") as MockClient:
        mock_client = _mock_collection_with_sparse()
        MockClient.return_value = mock_client

        from earthrise_rag.indexing.qdrant_store import QdrantStore
        from earthrise_rag.models import Chunk

        store = QdrantStore(
            "http://fake:6333", "test_collection", 10, sparse_embedder=FakeSparseEmbedder()
        )

        chunks = [
            Chunk(
                content="test content",
                content_hash="abc",
                source_type="book_text",
                content_type="concept",
            )
        ]
        vectors = [[0.1] * 10]

        store.upsert(chunks, vectors)

        call_args = mock_client.upsert.call_args
        points = call_args.kwargs.get("points") or call_args[1].get("points")
        vec_dict = points[0].vector
        assert "dense" in vec_dict
        assert "sparse" in vec_dict


def test_ensure_collection_rejects_missing_sparse_on_indexer_path():
    with patch("qdrant_client.QdrantClient") as MockClient:
        MockClient.return_value = _mock_collection_dense_only()

        from earthrise_rag.indexing.qdrant_store import QdrantStore

        with pytest.raises(ValueError, match="--recreate-collection"):
            QdrantStore("http://fake:6333", "test_collection", 10, create_if_missing=True)


def test_app_path_dense_only_collection_warns():
    with patch("qdrant_client.QdrantClient") as MockClient:
        MockClient.return_value = _mock_collection_dense_only()

        from earthrise_rag.indexing.qdrant_store import QdrantStore

        store = QdrantStore("http://fake:6333", "test_collection", 10, create_if_missing=False)

        assert store._has_sparse is False


def test_app_path_missing_collection_no_crash():
    with patch("qdrant_client.QdrantClient") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.get_collections.return_value.collections = []

        from earthrise_rag.indexing.qdrant_store import QdrantStore

        store = QdrantStore("http://fake:6333", "test_collection", 10, create_if_missing=False)

        assert store._has_sparse is False


def test_search_sparse_returns_empty_when_no_sparse():
    with patch("qdrant_client.QdrantClient") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.get_collections.return_value.collections = []

        from earthrise_rag.indexing.qdrant_store import QdrantStore

        store = QdrantStore("http://fake:6333", "test_collection", 10, create_if_missing=False)

        result = store.search_sparse("query text")
        assert result == []
        mock_client.query_points.assert_not_called()


def test_create_collection_uses_idf_modifier_when_required():
    with patch("qdrant_client.QdrantClient") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.get_collections.return_value.collections = []

        from earthrise_rag.indexing.qdrant_store import QdrantStore

        idf_embedder = FakeSparseEmbedder()
        idf_embedder.requires_idf = True

        QdrantStore(
            "http://fake:6333",
            "test_collection",
            10,
            create_if_missing=True,
            sparse_embedder=idf_embedder,
        )

        call_args = mock_client.create_collection.call_args
        sparse_config = call_args.kwargs.get("sparse_vectors_config") or call_args[1].get(
            "sparse_vectors_config"
        )
        sparse_params = sparse_config["sparse"]

        from qdrant_client.models import Modifier

        assert sparse_params.modifier == Modifier.IDF
