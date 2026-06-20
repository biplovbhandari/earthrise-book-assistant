from earthrise_rag.indexing.pipeline import IndexingPipeline
from earthrise_rag.models import Chunk, Document


class FakeParser:
    def __init__(self, content: str = "Real content here", title: str = "Test"):
        self._content = content
        self._title = title

    def parse(self, actual_path: str, source_path: str) -> Document:
        return Document(
            title=self._title,
            source_path=source_path,
            content=self._content,
            source_type="book_text",
        )


class FakeChunker:
    def chunk(self, document: Document) -> list[Chunk]:
        return [
            Chunk(
                content=document.content,
                content_hash="abc123",
                source_type="book_text",
                content_type="concept",
            )
        ]


class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 10 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 10

    def get_dimension(self) -> int:
        return 10


class FakeVectorStore:
    def __init__(self):
        self.deleted: list[str] = []
        self.upserted: list[tuple] = []

    def upsert(self, chunks, vectors):
        self.upserted.append((chunks, vectors))

    def delete_by_source(self, source_path: str):
        self.deleted.append(source_path)

    def search_dense(self, vector, top_k=10, filters=None):
        return []

    def search_sparse(self, text, top_k=10, filters=None):
        raise NotImplementedError

    def get_by_ids(self, ids):
        return []


def _make_pipeline(parser=None, chunker=None, store=None):
    store = store or FakeVectorStore()
    return IndexingPipeline(
        parsers={".md": parser or FakeParser()},
        chunkers={".md": chunker or FakeChunker()},
        embedder=FakeEmbedder(),
        vector_store=store,
    ), store


class TestPipeline:
    def test_skips_stub(self):
        parser = FakeParser(
            content="This chapter is under development and will be available later."
        )
        pipeline, store = _make_pipeline(parser=parser)
        result = pipeline.index_source("/tmp/stub.md", "book/stub.md")
        assert result.status == "skipped"
        assert result.chunks_indexed == 0
        assert len(store.upserted) == 0

    def test_builds_before_delete(self):
        call_order = []

        class TrackingStore(FakeVectorStore):
            def delete_by_source(self, source_path):
                call_order.append("delete")
                super().delete_by_source(source_path)

            def upsert(self, chunks, vectors):
                call_order.append("upsert")
                super().upsert(chunks, vectors)

        class TrackingEmbedder(FakeEmbedder):
            def embed_documents(self, texts):
                call_order.append("embed")
                return super().embed_documents(texts)

        store = TrackingStore()
        pipeline = IndexingPipeline(
            parsers={".md": FakeParser()},
            chunkers={".md": FakeChunker()},
            embedder=TrackingEmbedder(),
            vector_store=store,
        )
        pipeline.index_source("/tmp/test.md", "book/test.md")
        assert call_order == ["embed", "delete", "upsert"]
