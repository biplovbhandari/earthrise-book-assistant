import json

from fastapi.testclient import TestClient

from earthrise_rag.models import Chunk, ScoredChunk


def _make_scored_chunk(content="U-Net architecture", score=0.95):
    chunk = Chunk(
        content=content,
        content_hash="abc",
        source_type="book_text",
        content_type="concept",
        metadata={
            "source_path": "book/03_Segmentation/index.qmd",
            "chapter": "03",
            "section": "U-Net",
        },
    )
    return ScoredChunk(chunk=chunk, score=score, ranking_method="dense")


class FakeStreamingQueryPipeline:
    """Explicit fake with all required adapters for readiness checks."""

    def __init__(self):
        self._context_builder = object()
        self._llm_client = self  # self has chat_stream
        self._citation_builder = object()
        self.last_history = None

    def ask_stream(self, question, *, history=None, filters=None):
        self.last_history = history
        yield {"type": "meta", "citations": []}
        yield {"type": "token", "content": "Answer text."}
        yield {"type": "done"}

    def chat_stream(self, messages, temperature=0.3, max_tokens=1024):
        yield "test"


class FakeNonStreamingQueryPipeline:
    """Fake WITHOUT chat_stream -- must produce 503."""

    def __init__(self):
        self._context_builder = object()
        self._llm_client = _NoStreamLLM()
        self._citation_builder = object()


class _NoStreamLLM:
    def chat(self, messages, temperature=0.3, max_tokens=1024):
        return "answer"


class FakeNonCallableStreamQueryPipeline:
    """Fake with chat_stream as a non-callable attribute -- must produce 503."""

    def __init__(self):
        self._context_builder = object()
        self._llm_client = _NonCallableStreamLLM()
        self._citation_builder = object()


class _NonCallableStreamLLM:
    chat_stream = "not callable"

    def chat(self, messages, temperature=0.3, max_tokens=1024):
        return "answer"


class FakeIncompleteQueryPipeline:
    """Fake with _context_builder=None -- must produce 503."""

    def __init__(self):
        self._context_builder = None
        self._llm_client = object()
        self._citation_builder = object()


class FakeVectorStore:
    def __init__(self, count_val=100):
        self._count = count_val

    def count(self):
        return self._count


def _make_fake_pipelines(query=None, store=None):
    from api.dependencies import Pipelines

    return Pipelines(
        query=query or FakeStreamingQueryPipeline(),  # type: ignore[arg-type]
        vector_store=store or FakeVectorStore(),  # type: ignore[arg-type]
    )


def _create_client(monkeypatch, pipelines=None):
    fake = pipelines if pipelines is not None else _make_fake_pipelines()
    monkeypatch.setattr("api.main.create_pipelines", lambda config: fake)
    from api.main import app

    return TestClient(app)


class TestChatEndpoint:
    def test_streams_sse_events(self, monkeypatch):
        client = _create_client(monkeypatch)
        with client:
            resp = client.post("/chat", json={"question": "What is U-Net?"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line.removeprefix("data: "))
            for line in resp.text.strip().split("\n\n")
            if line.startswith("data:")
        ]
        types = [e["type"] for e in events]
        assert "meta" in types
        assert "token" in types
        assert "done" in types

    def test_no_streaming_returns_503(self, monkeypatch):
        pipelines = _make_fake_pipelines(query=FakeNonStreamingQueryPipeline())
        client = _create_client(monkeypatch, pipelines)
        with client:
            resp = client.post("/chat", json={"question": "Q?"})
        assert resp.status_code == 503

    def test_incomplete_adapters_returns_503(self, monkeypatch):
        pipelines = _make_fake_pipelines(query=FakeIncompleteQueryPipeline())
        client = _create_client(monkeypatch, pipelines)
        with client:
            resp = client.post("/chat", json={"question": "Q?"})
        assert resp.status_code == 503

    def test_empty_vector_store_returns_503(self, monkeypatch):
        pipelines = _make_fake_pipelines(store=FakeVectorStore(count_val=0))
        client = _create_client(monkeypatch, pipelines)
        with client:
            resp = client.post("/chat", json={"question": "Q?"})
        assert resp.status_code == 503

    def test_history_truncation(self, monkeypatch):
        fake_query = FakeStreamingQueryPipeline()
        pipelines = _make_fake_pipelines(query=fake_query)
        client = _create_client(monkeypatch, pipelines)
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(15)
        ]
        with client:
            resp = client.post("/chat", json={"question": "Q?", "history": history})
        assert resp.status_code == 200
        assert fake_query.last_history is not None
        assert len(fake_query.last_history) <= 10
        assert fake_query.last_history[0]["role"] == "user"

    def test_non_callable_chat_stream_returns_503(self, monkeypatch):
        pipelines = _make_fake_pipelines(query=FakeNonCallableStreamQueryPipeline())
        client = _create_client(monkeypatch, pipelines)
        with client:
            resp = client.post("/chat", json={"question": "Q?"})
        assert resp.status_code == 503

    def test_long_history_content_accepted(self, monkeypatch):
        fake_query = FakeStreamingQueryPipeline()
        pipelines = _make_fake_pipelines(query=fake_query)
        client = _create_client(monkeypatch, pipelines)
        history = [
            {"role": "user", "content": "short question"},
            {"role": "assistant", "content": "x" * 10000},
        ]
        with client:
            resp = client.post("/chat", json={"question": "Follow up?", "history": history})
        assert resp.status_code == 200

    def test_empty_question_returns_422(self, monkeypatch):
        client = _create_client(monkeypatch)
        with client:
            resp = client.post("/chat", json={"question": ""})
        assert resp.status_code == 422
