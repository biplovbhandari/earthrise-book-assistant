from earthrise_rag.models import Chunk, ScoredChunk
from earthrise_rag.models.citation import Citation
from earthrise_rag.query import QueryPipeline


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


class FakeStrategy:
    def __init__(self, results):
        self._results = results

    def retrieve(self, question, top_k, filters=None):
        return self._results[:top_k]


class FakeStreamingLLMClient:
    def chat(self, messages, temperature=0.3, max_tokens=1024):
        return "U-Net is a CNN [1]."

    def chat_stream(self, messages, temperature=0.3, max_tokens=1024):
        yield "U-Net "
        yield "is "
        yield "a CNN [1]."


class FakeEmptyStreamLLMClient:
    def chat(self, messages, temperature=0.3, max_tokens=1024):
        return ""

    def chat_stream(self, messages, temperature=0.3, max_tokens=1024):
        yield ""
        yield "  "


class FakeContextBuilder:
    def build(self, question, chunks, history=None):
        return [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"Q: {question}"},
        ]


class FakeCitationBuilder:
    def build(self, chunks):
        return [
            Citation(
                chunk_id=sc.chunk.id,
                source_path="book/ch1.qmd",
                chapter="01",
                section="Intro",
                url="/ch1.html",
            )
            for sc in chunks
        ]


class TestAskStream:
    def test_yields_meta_then_tokens_then_done(self):
        pipeline = QueryPipeline(
            strategy=FakeStrategy([_make_scored_chunk()]),
            context_builder=FakeContextBuilder(),
            llm_client=FakeStreamingLLMClient(),
            citation_builder=FakeCitationBuilder(),
            top_k=8,
        )
        events = list(pipeline.ask_stream("What is U-Net?"))
        types = [e["type"] for e in events]
        assert types[0] == "meta"
        assert types[-1] == "done"
        assert "token" in types

    def test_meta_has_citations(self):
        pipeline = QueryPipeline(
            strategy=FakeStrategy([_make_scored_chunk()]),
            context_builder=FakeContextBuilder(),
            llm_client=FakeStreamingLLMClient(),
            citation_builder=FakeCitationBuilder(),
            top_k=8,
        )
        events = list(pipeline.ask_stream("What is U-Net?"))
        meta = events[0]
        assert meta["type"] == "meta"
        assert len(meta["citations"]) == 1
        assert meta["citations"][0]["url"] == "/ch1.html"

    def test_zero_chunks_emits_meta_then_canned(self):
        pipeline = QueryPipeline(
            strategy=FakeStrategy([]),
            context_builder=FakeContextBuilder(),
            llm_client=FakeStreamingLLMClient(),
            citation_builder=FakeCitationBuilder(),
            top_k=8,
        )
        events = list(pipeline.ask_stream("Unknown topic?"))
        types = [e["type"] for e in events]
        assert types == ["meta", "token", "done"]
        assert events[0]["citations"] == []
        assert "No relevant information" in events[1]["content"]

    def test_empty_stream_yields_error(self):
        pipeline = QueryPipeline(
            strategy=FakeStrategy([_make_scored_chunk()]),
            context_builder=FakeContextBuilder(),
            llm_client=FakeEmptyStreamLLMClient(),
            citation_builder=FakeCitationBuilder(),
            top_k=8,
        )
        events = list(pipeline.ask_stream("Q?"))
        types = [e["type"] for e in events]
        assert "error" in types
        assert "done" not in types


class TestOpenAIStreamingAdapter:
    def test_chat_stream_yields_content_skips_empty_deltas(self, monkeypatch):
        """Verify the real adapter calls stream=True and filters empty deltas."""
        from unittest.mock import MagicMock

        from earthrise_rag.generation.llm_client import OpenAICompatibleClient

        chunk_with_content = MagicMock()
        chunk_with_content.choices = [MagicMock()]
        chunk_with_content.choices[0].delta.content = "Hello"

        chunk_empty_delta = MagicMock()
        chunk_empty_delta.choices = [MagicMock()]
        chunk_empty_delta.choices[0].delta.content = None

        chunk_no_choices = MagicMock()
        chunk_no_choices.choices = []

        chunk_final = MagicMock()
        chunk_final.choices = [MagicMock()]
        chunk_final.choices[0].delta.content = " world"

        fake_response = [chunk_with_content, chunk_empty_delta, chunk_no_choices, chunk_final]

        mock_openai = MagicMock()
        mock_openai.return_value.chat.completions.create.return_value = iter(fake_response)
        monkeypatch.setattr("openai.OpenAI", mock_openai)

        client = OpenAICompatibleClient(base_url="http://fake", api_key="fake", model="test-model")
        tokens = list(client.chat_stream([{"role": "user", "content": "Hi"}]))

        assert tokens == ["Hello", " world"]
        call_kwargs = mock_openai.return_value.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("stream") is True or call_kwargs[1].get("stream") is True


class TestBuildRetrievalQuery:
    def test_no_history(self):
        assert QueryPipeline._build_retrieval_query("What is X?", None) == "What is X?"

    def test_with_history(self):
        history = [
            {"role": "user", "content": "What is U-Net?"},
            {"role": "assistant", "content": "A CNN."},
        ]
        result = QueryPipeline._build_retrieval_query("Tell me more", history)
        assert result.startswith("Tell me more")
        assert "U-Net" in result

    def test_caps_at_500_chars(self):
        history = [{"role": "user", "content": "x" * 1000}]
        result = QueryPipeline._build_retrieval_query("Q", history)
        assert len(result) <= 504  # "Q " + 500 + margin

    def test_malformed_history(self):
        history = [{"bad_key": "value"}]
        result = QueryPipeline._build_retrieval_query("Q?", history)
        assert result == "Q?"
