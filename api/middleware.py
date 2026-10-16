from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


def client_ip(request: Request) -> str:
    """Return the real client IP, preferring the CF-Connecting-IP header set by Cloudflare.

    Behind the Cloudflare Tunnel every connection comes from the tunnel daemon, so the
    connection address is the same for all visitors and cannot tell them apart.
    The header is only trustworthy while Cloudflare is the sole route to the app, because a
    caller that connects directly can send any value.
    An empty header is ignored, and "unknown" is returned when the server reports no address.
    """
    return request.headers.get("CF-Connecting-IP") or (
        request.client.host if request.client else "unknown"
    )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP sliding window rate limiter with separate buckets for LLM and general endpoints.

    LLM endpoints (/ask, /chat) use the configured rate limit. Other API routes
    (/search, /analytics, /config, /eval, /health, /feedback) use a higher limit (4x) because
    they are cheap database reads or writes, not LLM calls. Static file requests pass through.
    """

    _LLM_PREFIXES = ("/ask", "/chat")
    _API_PREFIXES = (
        "/search",
        "/ask",
        "/chat",
        "/health",
        "/analytics",
        "/config",
        "/feedback",
        "/eval",
    )
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

        ip = client_ip(request)
        llm = self._is_llm(path)
        bucket = f"{ip}:llm" if llm else f"{ip}:general"
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
