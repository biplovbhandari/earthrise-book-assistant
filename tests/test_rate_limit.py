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

    @app.get("/analytics/overview")
    async def analytics():
        return {"total": 42}

    @app.post("/feedback")
    async def feedback():
        """Stand-in for the feedback endpoint."""
        return {"ok": True}

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

    def test_rejects_llm_requests_over_limit(self):
        """LLM endpoints (/chat, /ask) hit the configured limit."""
        client = TestClient(_create_app(max_requests=3))
        for _ in range(3):
            assert client.post("/chat").status_code == 200

        resp = client.post("/chat")
        assert resp.status_code == 429
        assert "Too many requests" in resp.json()["detail"]

    def test_general_endpoints_get_higher_limit(self):
        """Non-LLM API routes get 4x the configured limit."""
        client = TestClient(_create_app(max_requests=2))
        # General limit = 2 * 4 = 8
        for _ in range(8):
            assert client.get("/health").status_code == 200

        resp = client.get("/health")
        assert resp.status_code == 429

    def test_static_routes_bypass_limit(self):
        """Non-API routes should not be rate-limited."""
        client = TestClient(_create_app(max_requests=2))
        # Exhaust LLM bucket
        client.post("/chat")
        client.post("/chat")
        assert client.post("/chat").status_code == 429

        resp = client.get("/static-page")
        assert resp.status_code == 200

    def test_llm_and_general_have_separate_buckets(self):
        """Exhausting the LLM bucket does not affect general endpoints."""
        client = TestClient(_create_app(max_requests=2))
        client.post("/chat")
        client.post("/chat")
        assert client.post("/chat").status_code == 429

        assert client.get("/health").status_code == 200
        assert client.get("/analytics/overview").status_code == 200

    def test_analytics_uses_general_bucket(self):
        """Analytics endpoints share the general bucket with the 4x multiplier."""
        client = TestClient(_create_app(max_requests=1))
        # General limit = 1 * 4 = 4
        for _ in range(4):
            assert client.get("/analytics/overview").status_code == 200

        assert client.get("/analytics/overview").status_code == 429

    def test_feedback_uses_general_bucket(self):
        """Feedback shares the general bucket, separate from LLM."""
        client = TestClient(_create_app(max_requests=1))
        # General limit = 1 * 4 = 4
        for _ in range(4):
            assert client.post("/feedback").status_code == 200

        assert client.post("/feedback").status_code == 429

    def test_window_expiry_resets_count(self):
        """Requests outside the window should not count."""
        client = TestClient(_create_app(max_requests=2))
        client.post("/chat")
        client.post("/chat")
        assert client.post("/chat").status_code == 429

        with patch("api.middleware.time") as mock_time:
            mock_time.monotonic.return_value = 1e9
            resp = client.post("/chat")
            assert resp.status_code == 200
