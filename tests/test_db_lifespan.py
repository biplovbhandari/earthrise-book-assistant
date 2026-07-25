"""Tests for database graceful degradation in the FastAPI lifespan."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_no_db(monkeypatch):
    """TestClient with DATABASE_URL empty (DB disabled), lifespan executed."""
    monkeypatch.setenv("DATABASE_URL", "")
    from earthrise_rag.config import get_settings

    get_settings.cache_clear()
    from api.main import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def test_health_shows_database_disabled_when_no_url(client_no_db):
    """With empty DATABASE_URL, health reports database: disabled and RAG still works."""
    resp = client_no_db.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["database"] == "disabled"
    assert data["status"] == "ok"
