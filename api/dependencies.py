from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from earthrise_rag.config import Settings

if TYPE_CHECKING:
    from earthrise_rag.interfaces import VectorStore
    from earthrise_rag.query import QueryPipeline

logger = logging.getLogger(__name__)


@dataclass
class Pipelines:
    """Container for wired pipeline objects, set on app.state at startup."""

    indexing: object | None = field(default=None)
    query: QueryPipeline | None = field(default=None)
    vector_store: VectorStore | None = field(default=None)


def _create_embedder(config: Settings):
    from earthrise_rag.indexing.embedder import LocalEmbeddingModel

    if config.embedding_provider == "local":
        return LocalEmbeddingModel(config.embedding_model_name, config.hf_home)
    raise ValueError(f"Unknown embedding_provider: {config.embedding_provider}")


def _create_vector_store(config: Settings, dense_dim: int, create_if_missing: bool):
    from earthrise_rag.indexing.qdrant_store import QdrantStore

    if config.vector_store_provider == "qdrant":
        return QdrantStore(
            config.qdrant_url,
            config.qdrant_collection,
            dense_dim,
            create_if_missing=create_if_missing,
        )
    raise ValueError(f"Unknown vector_store_provider: {config.vector_store_provider}")


def _create_parsers() -> dict:
    from earthrise_rag.indexing.parsers import BibParser, MarkdownParser, NotebookParser

    return {
        ".md": MarkdownParser(),
        ".qmd": MarkdownParser(),
        ".ipynb": NotebookParser(),
        ".bib": BibParser(),
    }


def _create_chunkers() -> dict:
    from earthrise_rag.indexing.chunkers import BibChunker, NotebookChunker, SectionChunker

    return {
        ".md": SectionChunker(),
        ".qmd": SectionChunker(),
        ".ipynb": NotebookChunker(),
        ".bib": BibChunker(),
    }


def _create_reranker(config: Settings):
    from earthrise_rag.retrieval.rerankers import NoOpReranker

    if config.reranker_provider == "noop":
        return NoOpReranker()
    raise ValueError(f"Unknown reranker_provider: {config.reranker_provider}")


def _create_retrieval_strategy(config: Settings, embedder, store, reranker):
    from earthrise_rag.retrieval.strategies import DenseStrategy

    if config.retrieval_strategy == "dense":
        return DenseStrategy(embedder, store, reranker)
    if config.retrieval_strategy == "hybrid":
        raise NotImplementedError("Hybrid search hasn't been implemented yet.")
    raise ValueError(f"Unknown retrieval_strategy: {config.retrieval_strategy}")


def _create_llm_client(config: Settings):
    """Build the LLM client from config (provider pattern)."""
    from earthrise_rag.generation.llm_client import OpenAICompatibleClient

    if config.llm_provider == "openai_compatible":
        return OpenAICompatibleClient(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key.get_secret_value(),
            model=config.llm_model,
            timeout=config.llm_timeout_seconds,
        )
    raise ValueError(f"Unknown llm_provider: {config.llm_provider}")


def _create_context_builder():
    """Build the context builder (direct pattern, single adapter)."""
    from earthrise_rag.generation.context_builder import DefaultContextBuilder

    return DefaultContextBuilder()


def _create_citation_builder():
    """Build the citation builder (direct pattern, single adapter)."""
    from earthrise_rag.citations.citation_builder import DefaultCitationBuilder

    return DefaultCitationBuilder()


def create_indexing_pipeline(config: Settings):
    """Build the indexing pipeline for CLI use (create_if_missing=True)."""
    from earthrise_rag.indexing.pipeline import IndexingPipeline

    embedder = _create_embedder(config)
    store = _create_vector_store(config, embedder.get_dimension(), create_if_missing=True)

    return IndexingPipeline(
        parsers=_create_parsers(),
        chunkers=_create_chunkers(),
        embedder=embedder,
        vector_store=store,
        commit_sha=config.book_commit_sha,
    )


def create_pipelines(config: Settings) -> Pipelines:
    """Build the query-side pipelines for FastAPI startup (create_if_missing=False)."""
    from earthrise_rag.query import QueryPipeline

    embedder = _create_embedder(config)
    store = _create_vector_store(config, embedder.get_dimension(), create_if_missing=False)
    reranker = _create_reranker(config)
    strategy = _create_retrieval_strategy(config, embedder, store, reranker)

    context_builder = None
    llm_client = None
    citation_builder = None
    try:
        context_builder = _create_context_builder()
        llm_client = _create_llm_client(config)
        citation_builder = _create_citation_builder()
    except Exception:
        logger.warning(
            "LLM client creation failed; /ask will return 503 but /search is unaffected",
            exc_info=True,
        )

    return Pipelines(
        query=QueryPipeline(
            strategy=strategy,
            context_builder=context_builder,
            llm_client=llm_client,
            citation_builder=citation_builder,
            top_k=config.retrieval_top_k,
        ),
        vector_store=store,
    )
