from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from conftest import create_test_client, make_scored_chunk

from earthrise_rag.models.answer import Answer
from earthrise_rag.models.citation import Citation


def _make_answer(chunks=None):
    if chunks is None:
        chunks = [make_scored_chunk()]
    return Answer(
        answer="U-Net is a convolutional neural network [1].",
        sources=chunks,
        citations=[
            Citation(
                chunk_id=sc.chunk.id,
                source_path=sc.chunk.metadata.get("source_path", ""),
                chapter=sc.chunk.metadata.get("chapter", ""),
                section=sc.chunk.metadata.get("section", ""),
            )
            for sc in chunks
        ],
    )


class _FakeQueryPipeline:
    def __init__(self, ask_result):
        self._context_builder = object()
        self._llm_client = self
        self._citation_builder = object()
        self._ask_result = ask_result

    def ask(self, question, filters=None, *, history=None):
        return self._ask_result

    def chat_stream(self, messages, temperature=0.3, max_tokens=1024):
        yield "test"


class _FakeFailingQueryPipeline(_FakeQueryPipeline):
    def ask(self, question, filters=None, *, history=None):
        raise Exception("LLM timeout")


def _make_fake_pipelines(ask_result=None, count=100):
    from api.dependencies import Pipelines

    query = _FakeQueryPipeline(ask_result or _make_answer())
    store = type("FakeStore", (), {"count": lambda self: count})()

    return Pipelines(query=query, vector_store=store)  # type: ignore[arg-type]


class TestAsk:
    def test_valid_query_returns_answer(self, monkeypatch):
        client = create_test_client(monkeypatch, _make_fake_pipelines())
        with client:
            resp = client.post("/ask", json={"question": "What is U-Net?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "sources" in body
        assert "citations" in body
        assert len(body["sources"]) == 1
        assert len(body["citations"]) == 1

    def test_pipelines_none_returns_503(self, monkeypatch):
        monkeypatch.setattr(
            "api.main.create_pipelines",
            MagicMock(side_effect=Exception("Qdrant down")),
        )
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/ask", json={"question": "test"})
        assert resp.status_code == 503

    def test_llm_failure_returns_503(self, monkeypatch):
        from api.dependencies import Pipelines

        query = _FakeFailingQueryPipeline(_make_answer())
        store = type("FakeStore", (), {"count": lambda self: 100})()
        pipelines = Pipelines(query=query, vector_store=store)  # type: ignore[arg-type]
        client = create_test_client(monkeypatch, pipelines)
        with client:
            resp = client.post("/ask", json={"question": "test"})
        assert resp.status_code == 503
