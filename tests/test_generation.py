from conftest import FakeCitationBuilder, FakeContextBuilder, FakeStrategy, make_scored_chunk

from earthrise_rag.query import QueryPipeline


class FakeLLMClient:
    """Fake LLM that returns a canned answer."""

    def chat(self, messages, temperature=0.3, max_tokens=1024):
        """Return a canned answer string."""
        return "U-Net is a convolutional neural network [1]."

    def chat_stream(self, messages, temperature=0.3, max_tokens=1024):
        """Yield the canned answer as a single token."""
        yield self.chat(messages, temperature, max_tokens)


def test_ask_compose_chain():
    canned = [make_scored_chunk()]
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
