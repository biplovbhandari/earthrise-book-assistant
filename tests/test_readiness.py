from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize("route", ["/ask", "/search", "/chat"])
def test_pipelines_none_returns_503(monkeypatch, route):
    """When startup pipeline init fails, every RAG route degrades to 503, not 500.

    All three routes share ``check_retrieval_ready``; its pipelines-is-None branch
    is the single guard exercised here across every route.
    """
    monkeypatch.setattr(
        "api.main.create_pipelines",
        MagicMock(side_effect=Exception("Qdrant down")),
    )
    from api.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(route, json={"question": "test"})
    assert resp.status_code == 503
