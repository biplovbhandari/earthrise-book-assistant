from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from earthrise_rag.logging_config import request_id as request_id_var


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


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assign a unique request ID to each request for log correlation.

    Reads X-Request-ID from the incoming header (Cloudflare or upstream proxy
    may set one) or generates a new UUID4. Stores it in a ContextVar for the
    logging filter and adds it to the response headers.
    """

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request_id_var.set(rid)
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


_access_logger = logging.getLogger("access")

_API_PREFIXES = (
    "/search",
    "/ask",
    "/chat",
    "/health",
    "/analytics",
    "/config",
    "/feedback",
    "/eval",
    "/metrics",
)


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Log one structured line per API request.

    Excludes static file requests (book pages, dashboard assets) to keep
    access logs focused on API traffic.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith(_API_PREFIXES):
            return await call_next(request)

        start = time.monotonic()
        response = await call_next(request)
        latency_ms = int((time.monotonic() - start) * 1000)

        _access_logger.info(
            "%s %s %d %dms",
            request.method,
            path,
            response.status_code,
            latency_ms,
            extra={
                "method": request.method,
                "path": path,
                "status": response.status_code,
                "latency_ms": latency_ms,
                "client_ip": client_ip(request),
            },
        )
        return response


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
