from earthrise_rag.models import Chunk, ScoredChunk
from earthrise_rag.models.citation import Citation
from earthrise_rag.query import QueryPipeline


def _make_scored_chunk(content: str = "U-Net architecture", score: float = 0.95) -> ScoredChunk:
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


class FakeStrategy:
    def __init__(self, results):
        self._results = results

    def retrieve(self, question, top_k, filters=None):
        return self._results[:top_k]


class FakeContextBuilder:
    def build(self, question, chunks):
        return [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"Context: ...\n\nQuestion: {question}"},
        ]


class FakeLLMClient:
    def chat(self, messages, temperature=0.3, max_tokens=1024):
        return "U-Net is a convolutional neural network [1]."


class FakeCitationBuilder:
    def build(self, chunks):
        return [
            Citation(
                chunk_id=sc.chunk.id,
                source_path=sc.chunk.metadata.get("source_path", ""),
                chapter=sc.chunk.metadata.get("chapter", ""),
                section=sc.chunk.metadata.get("section", ""),
            )
            for sc in chunks
        ]


def test_ask_compose_chain():
    canned = [_make_scored_chunk()]
    pipeline = QueryPipeline(
        strategy=FakeStrategy(canned),
        context_builder=FakeContextBuilder(),
        llm_client=FakeLLMClient(),
        citation_builder=FakeCitationBuilder(),
        top_k=8,
    )

    result = pipeline.ask("What is U-Net?")

    assert result.answer == "U-Net is a convolutional neural network [1]."
    assert len(result.sources) == 1
    assert result.sources[0].chunk.content == "U-Net architecture"
    assert len(result.citations) == 1
    assert result.citations[0].chapter == "03"
    assert result.citations[0].section == "U-Net"


def test_ask_zero_chunks_returns_canned_answer():
    pipeline = QueryPipeline(
        strategy=FakeStrategy([]),
        context_builder=FakeContextBuilder(),
        llm_client=FakeLLMClient(),
        citation_builder=FakeCitationBuilder(),
        top_k=8,
    )

    result = pipeline.ask("What is something not in the book?")

    assert "No relevant information found" in result.answer
    assert result.sources == []
    assert result.citations == []
