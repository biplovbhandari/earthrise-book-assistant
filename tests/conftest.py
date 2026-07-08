"""Shared test fixtures, factories, and fakes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from earthrise_rag.models import Chunk, ScoredChunk
from earthrise_rag.models.citation import Citation


@pytest.fixture
def client():
    """Create a bare TestClient for the FastAPI app."""
    from api.main import app

    return TestClient(app)


def make_scored_chunk(
    content="U-Net architecture",
    score=0.95,
    method="dense",
    metadata=None,
):
    """Create a ScoredChunk for testing."""
    chunk = Chunk(
        content=content,
        content_hash="abc",
        source_type="book_text",
        content_type="concept",
        metadata=metadata
        or {
            "source_path": "book/03_Segmentation/index.qmd",
            "chapter": "03",
            "section": "U-Net",
        },
    )
    return ScoredChunk(chunk=chunk, score=score, ranking_method=method)


class FakeStrategy:
    """Canned retrieval strategy for testing."""

    def __init__(self, results):
        self._results = results

    def retrieve(self, question, top_k, filters=None):
        """Return canned results up to top_k."""
        return self._results[:top_k]


class FakeContextBuilder:
    """Canned context builder for testing."""

    def build(self, question, chunks, history=None):
        """Return a fixed two-message context."""
        return [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"Q: {question}"},
        ]


class FakeCitationBuilder:
    """Canned citation builder for testing."""

    def build(self, chunks):
        """Return one citation per chunk with a fixed URL."""
        citations = []
        for sc in chunks:
            chapter = sc.chunk.metadata.get("chapter", "")
            section = sc.chunk.metadata.get("section", "")
            if chapter and section:
                display_label = f"{chapter} - {section}"
            else:
                display_label = chapter or section or "Source"
            citations.append(
                Citation(
                    chunk_id=sc.chunk.id,
                    source_path=sc.chunk.metadata.get("source_path", ""),
                    chapter=chapter,
                    section=section,
                    url="/ch1.html",
                    display_label=display_label,
                )
            )
        return citations


def create_test_client(monkeypatch, pipelines=None):
    """Create a TestClient with mocked pipelines."""
    from api.main import app

    if pipelines is not None:
        monkeypatch.setattr("api.main.create_pipelines", lambda config: pipelines)

    return TestClient(app)
