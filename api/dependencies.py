from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from earthrise_rag.config import Settings

if TYPE_CHECKING:
    from earthrise_rag.interfaces import VectorStore
    from earthrise_rag.query import QueryPipeline


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

    return Pipelines(
        query=QueryPipeline(strategy),
        vector_store=store,
    )
