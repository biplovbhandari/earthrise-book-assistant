from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from conftest import create_test_client, make_scored_chunk


def _make_fake_pipelines(search_results=None, count=100):
    from api.dependencies import Pipelines

    query = MagicMock()
    query.search.return_value = search_results or [make_scored_chunk()]

    store = MagicMock()
    store.count.return_value = count

    return Pipelines(query=query, vector_store=store)  # type: ignore[arg-type]


class TestSearch:
    def test_valid_query_returns_chunks(self, monkeypatch):
        client = create_test_client(monkeypatch, _make_fake_pipelines())
        with client:
            resp = client.post("/search", json={"question": "What is U-Net?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "chunks" in body
        assert len(body["chunks"]) == 1
        assert body["chunks"][0]["chunk"]["content"] == "U-Net architecture"

    def test_dotted_filter_key_rejected(self, monkeypatch):
        client = create_test_client(monkeypatch, _make_fake_pipelines())
        with client:
            resp = client.post(
                "/search",
                json={"question": "test", "filters": {"metadata.chapter": "03"}},
            )
        assert resp.status_code == 422

    def test_pipelines_none_returns_503(self, monkeypatch):
        monkeypatch.setattr(
            "api.main.create_pipelines",
            MagicMock(side_effect=Exception("Qdrant down")),
        )
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/search", json={"question": "test"})
        assert resp.status_code == 503

    def test_transient_qdrant_failure_returns_503(self, monkeypatch):
        pipelines = _make_fake_pipelines()
        pipelines.query.search.side_effect = Exception("connection refused")  # type: ignore[union-attr]
        client = create_test_client(monkeypatch, pipelines)
        with client:
            resp = client.post("/search", json={"question": "test"})
        assert resp.status_code == 503
