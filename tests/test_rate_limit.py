"""Tests for the rate limiting middleware."""

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware import RateLimitMiddleware


def _create_app(max_requests: int = 5) -> FastAPI:
    """Create a minimal app with rate limiting for testing."""
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, max_requests=max_requests, window_seconds=60)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/chat")
    async def chat():
        return {"response": "hello"}

    @app.get("/static-page")
    async def static_page():
        return {"page": "content"}

    return app


class TestRateLimitMiddleware:
    def test_allows_requests_under_limit(self):
        """Requests under the limit should pass through."""
        client = TestClient(_create_app(max_requests=5))
        for _ in range(5):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_rejects_requests_over_limit(self):
        """The request exceeding the limit should get 429."""
        client = TestClient(_create_app(max_requests=3))
        for _ in range(3):
            resp = client.get("/health")
            assert resp.status_code == 200

        resp = client.get("/health")
        assert resp.status_code == 429
        assert "Too many requests" in resp.json()["detail"]

    def test_static_routes_bypass_limit(self):
        """Non-API routes should not be rate-limited."""
        client = TestClient(_create_app(max_requests=2))
        for _ in range(2):
            client.get("/health")

        resp = client.get("/health")
        assert resp.status_code == 429

        resp = client.get("/static-page")
        assert resp.status_code == 200

    def test_limit_applies_to_all_api_routes(self):
        """Rate limit is shared across API routes for the same IP."""
        client = TestClient(_create_app(max_requests=3))
        client.get("/health")
        client.post("/chat")
        client.get("/health")

        resp = client.post("/chat")
        assert resp.status_code == 429

    def test_window_expiry_resets_count(self):
        """Requests outside the window should not count."""
        client = TestClient(_create_app(max_requests=2))
        client.get("/health")
        client.get("/health")

        resp = client.get("/health")
        assert resp.status_code == 429

        with patch("api.middleware.time") as mock_time:
            mock_time.monotonic.return_value = 1e9
            resp = client.get("/health")
            assert resp.status_code == 200
