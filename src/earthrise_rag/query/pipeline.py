from __future__ import annotations

from typing import Any

from earthrise_rag.interfaces import (
    CitationBuilder,
    ContextBuilder,
    LLMClient,
    RetrievalStrategy,
)
from earthrise_rag.models.answer import Answer
from earthrise_rag.models.scored_chunk import ScoredChunk


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

    def ask(self, question: str, filters: dict[str, Any] | None = None) -> Answer:
        """Retrieve chunks, build context, call LLM, and return a cited answer.

        Args:
            question: The user's natural language question.
            filters: Optional metadata filters for retrieval.

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

        chunks = self._strategy.retrieve(question, self._top_k, filters)

        if not chunks:
            return Answer(
                answer="No relevant information found in the book for this question.",
                sources=[],
                citations=[],
            )

        messages = self._context_builder.build(question, chunks)
        answer_text = self._llm_client.chat(messages)
        citations = self._citation_builder.build(chunks)

        return Answer(answer=answer_text, sources=chunks, citations=citations)
