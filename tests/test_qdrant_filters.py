from unittest.mock import MagicMock, patch


def test_filter_key_mapping():
    with patch("qdrant_client.QdrantClient") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        mock_collection_info = MagicMock()
        mock_vectors = {"dense": MagicMock(size=10, distance=MagicMock(__eq__=lambda s, o: True))}
        mock_collection_info.config.params.vectors = mock_vectors
        mock_client.get_collections.return_value.collections = [MagicMock(name="test_collection")]
        mock_client.get_collection.return_value = mock_collection_info

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
