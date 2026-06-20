from __future__ import annotations

from dataclasses import dataclass, field

from earthrise_rag.config import Settings


@dataclass
class Pipelines:
    indexing: object | None = field(default=None)
    query: object | None = field(default=None)


def _create_embedder(config: Settings):
    from earthrise_rag.indexing.embedder import LocalEmbeddingModel

    if config.embedding_provider == "local":
        return LocalEmbeddingModel(config.embedding_model_name, config.hf_home)
    raise ValueError(f"Unknown embedding_provider: {config.embedding_provider}")


def _create_vector_store(config: Settings, dense_dim: int):
    from earthrise_rag.indexing.qdrant_store import QdrantStore

    if config.vector_store_provider == "qdrant":
        return QdrantStore(config.qdrant_url, config.qdrant_collection, dense_dim)
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


def create_pipelines(config: Settings) -> Pipelines:
    from earthrise_rag.indexing.pipeline import IndexingPipeline

    embedder = _create_embedder(config)
    store = _create_vector_store(config, embedder.get_dimension())

    indexing = IndexingPipeline(
        parsers=_create_parsers(),
        chunkers=_create_chunkers(),
        embedder=embedder,
        vector_store=store,
        commit_sha=config.book_commit_sha,
    )

    return Pipelines(indexing=indexing, query=None)
