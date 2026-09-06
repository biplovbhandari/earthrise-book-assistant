"""Tests for LLM concurrency control via semaphore."""

import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient


def _create_app(max_concurrent: int = 1, delay: float = 0.2) -> FastAPI:
    """Create a minimal app with a semaphore-gated endpoint."""
    app = FastAPI()
    app.state.llm_semaphore = threading.Semaphore(max_concurrent)

    @app.post("/chat")
    def chat(request: Request):
        sem = request.app.state.llm_semaphore
        sem.acquire()
        try:
            time.sleep(delay)
        finally:
            sem.release()
        return JSONResponse({"response": "done"})

    @app.post("/search")
    def search():
        return JSONResponse({"results": []})

    return app


class TestLLMConcurrency:
    def test_single_request_proceeds(self):
        """A single request should acquire and release the semaphore."""
        client = TestClient(_create_app(max_concurrent=1, delay=0.05))
        resp = client.post("/chat")
        assert resp.status_code == 200

    def test_concurrent_requests_are_serialized(self):
        """With max_concurrent=1, two requests should not overlap."""
        app = _create_app(max_concurrent=1, delay=0.15)
        client = TestClient(app)
        results = []

        def make_request(label):
            start = time.monotonic()
            client.post("/chat")
            elapsed = time.monotonic() - start
            results.append((label, elapsed))

        t1 = threading.Thread(target=make_request, args=("a",))
        t2 = threading.Thread(target=make_request, args=("b",))
        t1.start()
        time.sleep(0.02)
        t2.start()
        t1.join()
        t2.join()

        first = min(results, key=lambda r: r[1])
        second = max(results, key=lambda r: r[1])
        assert second[1] > first[1] * 1.5

    def test_search_not_gated(self):
        """Non-LLM endpoints should not be blocked by the semaphore."""
        app = _create_app(max_concurrent=1, delay=0.15)
        client = TestClient(app)

        app.state.llm_semaphore.acquire()
        try:
            resp = client.post("/search")
            assert resp.status_code == 200
        finally:
            app.state.llm_semaphore.release()

    def test_semaphore_released_on_error(self):
        """The semaphore should be released even if the handler raises."""
        app = FastAPI()
        app.state.llm_semaphore = threading.Semaphore(1)

        @app.post("/chat")
        def chat(request: Request):
            sem = request.app.state.llm_semaphore
            sem.acquire()
            try:
                raise RuntimeError("LLM failed")
            finally:
                sem.release()

        client = TestClient(app, raise_server_exceptions=False)
        client.post("/chat")

        assert app.state.llm_semaphore.acquire(timeout=0.1)
        app.state.llm_semaphore.release()
