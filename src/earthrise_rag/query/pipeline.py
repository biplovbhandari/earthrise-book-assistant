from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from earthrise_rag.interfaces import (
    CitationBuilder,
    ContextBuilder,
    LLMClient,
    RetrievalStrategy,
)
from earthrise_rag.models.answer import Answer
from earthrise_rag.models.scored_chunk import ScoredChunk

_MAX_AUGMENT_CHARS = 500


class QueryPipeline:
    """Composition root for query operations (search and ask)."""

    def __init__(
        self,
        strategy: RetrievalStrategy,
        context_builder: ContextBuilder | None = None,
        llm_client: LLMClient | None = None,
        citation_builder: CitationBuilder | None = None,
        top_k: int = 8,
    ) -> None:
        """Initialize the query pipeline.

        Args:
            strategy: Retrieval strategy for finding relevant chunks.
            context_builder: Assembles chunks into LLM messages (None disables ask).
            llm_client: Generates answers from messages (None disables ask).
            citation_builder: Extracts citation metadata from chunks (None disables ask).
            top_k: Number of chunks to retrieve for ask() (from settings).
        """
        self._strategy = strategy
        self._context_builder = context_builder
        self._llm_client = llm_client
        self._citation_builder = citation_builder
        self._top_k = top_k

    def search(
        self,
        question: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Search for relevant chunks using the configured retrieval strategy.

        Args:
            question: Natural language query.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters.

        Returns:
            Ranked list of scored chunks.
        """
        return self._strategy.retrieve(question, top_k, filters)

    @staticmethod
    def _build_retrieval_query(
        question: str,
        history: list[dict[str, str]] | None,
    ) -> str:
        """Augment query with context from the last user message in history.

        Current question comes first so it survives embedding model truncation.
        Uses defensive dict access for direct pipeline callers that may pass
        malformed history dicts.

        Args:
            question: The current user question.
            history: Optional conversation history as a list of role/content dicts.

        Returns:
            The question, optionally appended with the most recent prior user message.
        """
        if not history:
            return question
        for msg in reversed(history):
            if msg.get("role") == "user":
                prev = str(msg.get("content", ""))[:_MAX_AUGMENT_CHARS]
                return f"{question} {prev}"
        return question

    def ask(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        *,
        history: list[dict[str, str]] | None = None,
    ) -> Answer:
        """Retrieve chunks, build context, call LLM, and return a cited answer.

        Args:
            question: The user's natural language question.
            filters: Optional metadata filters for retrieval.
            history: Optional conversation history as a list of role/content dicts.

        Returns:
            Answer with generated text, source chunks, and citations.

        Raises:
            RuntimeError: If generation adapters are not configured.
        """
        if (
            self._context_builder is None
            or self._llm_client is None
            or self._citation_builder is None
        ):
            raise RuntimeError(
                "Generation not configured"
                " - context_builder, llm_client, and citation_builder are required"
            )

        retrieval_query = self._build_retrieval_query(question, history)
        chunks = self._strategy.retrieve(retrieval_query, self._top_k, filters)

        if not chunks:
            return Answer(
                answer="No relevant information found in the book for this question.",
                sources=[],
                citations=[],
            )

        messages = self._context_builder.build(question, chunks, history=history)
        answer_text = self._llm_client.chat(messages)
        citations = self._citation_builder.build(chunks)

        return Answer(answer=answer_text, sources=chunks, citations=citations)

    def ask_stream(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream a RAG answer as SSE-compatible events.

        Yields dicts with a ``type`` key: ``meta`` (citations), ``token``
        (text chunk), ``error``, or ``done``.  The ``meta`` event is always
        emitted first, even for zero-chunk responses.

        Readiness is NOT checked here -- the route handler must verify all
        adapters before constructing ``StreamingResponse``.

        Args:
            question: The user's natural language question.
            history: Optional conversation history as a list of role/content dicts.
            filters: Optional metadata filters for retrieval.

        Yields:
            Event dicts with a ``type`` key and type-specific fields.

        Raises:
            RuntimeError: If generation adapters are not configured.
        """
        if (
            self._context_builder is None
            or self._llm_client is None
            or self._citation_builder is None
        ):
            raise RuntimeError(
                "Generation not configured"
                " - context_builder, llm_client, and citation_builder are required"
            )

        retrieval_query = self._build_retrieval_query(question, history)
        chunks = self._strategy.retrieve(retrieval_query, self._top_k, filters)

        if not chunks:
            yield {"type": "meta", "citations": []}
            yield {
                "type": "token",
                "content": "No relevant information found in the book for this question.",
            }
            yield {"type": "done"}
            return

        citations = self._citation_builder.build(chunks)
        yield {"type": "meta", "citations": [c.model_dump() for c in citations]}

        messages = self._context_builder.build(question, chunks, history=history)

        has_content = False
        for token in self._llm_client.chat_stream(messages):
            yield {"type": "token", "content": token}
            if token.strip():
                has_content = True

        if not has_content:
            yield {"type": "error", "message": "The model returned an empty response."}
            return

        yield {"type": "done"}
