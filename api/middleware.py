from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP sliding window rate limiter with separate buckets for LLM and general endpoints.

    LLM endpoints (/ask, /chat) use the configured rate limit. Other API routes
    (/search, /analytics, /health, /feedback) use a higher limit (4x) because they
    are cheap database reads or writes, not LLM calls. Static file requests pass through.
    """

    _LLM_PREFIXES = ("/ask", "/chat")
    _API_PREFIXES = ("/search", "/ask", "/chat", "/health", "/analytics", "/feedback")
    _GENERAL_MULTIPLIER = 4

    def __init__(self, app, max_requests: int = 30, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _is_llm(self, path: str) -> bool:
        """True for endpoints that call the LLM and need the stricter limit."""
        return path.startswith(self._LLM_PREFIXES)

    async def dispatch(self, request: Request, call_next):
        """Check rate limit for API routes, pass through static requests."""
        path = request.url.path
        if not path.startswith(self._API_PREFIXES):
            return await call_next(request)

        client_ip = request.headers.get(
            "cf-connecting-ip", request.client.host if request.client else "unknown"
        )
        llm = self._is_llm(path)
        bucket = f"{client_ip}:llm" if llm else f"{client_ip}:general"
        limit = self.max_requests if llm else self.max_requests * self._GENERAL_MULTIPLIER

        now = time.monotonic()
        cutoff = now - self.window_seconds

        timestamps = self._requests[bucket]
        self._requests[bucket] = [t for t in timestamps if t > cutoff]

        if len(self._requests[bucket]) >= limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
            )

        self._requests[bucket].append(now)
        return await call_next(request)
