from __future__ import annotations

import logging
from typing import Any

from earthrise_rag.models.chunk import Chunk
from earthrise_rag.models.scored_chunk import ScoredChunk

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100


class QdrantStore:
    def __init__(self, url: str, collection_name: str, dense_dim: int) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(url=url)
        self._collection = collection_name
        self._dense_dim = dense_dim

        self._ensure_collection(Distance, VectorParams)

    def _ensure_collection(self, Distance, VectorParams) -> None:
        collections = [c.name for c in self._client.get_collections().collections]

        if self._collection in collections:
            info = self._client.get_collection(self._collection)
            vectors_config = info.config.params.vectors

            if isinstance(vectors_config, dict) and "dense" in vectors_config:
                existing = vectors_config["dense"]
                if existing.size != self._dense_dim:
                    raise ValueError(
                        f"Collection '{self._collection}' has dense dim {existing.size}, "
                        f"expected {self._dense_dim}. Delete the collection or use a matching model."
                    )
                if existing.distance != Distance.COSINE:
                    raise ValueError(
                        f"Collection '{self._collection}' uses distance {existing.distance}, "
                        f"expected COSINE."
                    )
                logger.info("Collection '%s' exists with matching config.", self._collection)
            else:
                raise ValueError(
                    f"Collection '{self._collection}' exists but has no 'dense' named vector."
                )
        else:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    "dense": VectorParams(size=self._dense_dim, distance=Distance.COSINE)
                },
            )
            logger.info(
                "Created collection '%s' (dense_dim=%d).", self._collection, self._dense_dim
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must have equal length"
            )

        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=chunk.id,
                vector={"dense": vector},
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        for i in range(0, len(points), _BATCH_SIZE):
            batch = points[i : i + _BATCH_SIZE]
            self._client.upsert(collection_name=self._collection, points=batch)

    def search_dense(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        _TOP_LEVEL_FIELDS = {"source_type", "content_type", "chunk_type"}

        query_filter = None
        if filters:
            conditions = [
                FieldCondition(
                    key=k if k in _TOP_LEVEL_FIELDS else f"metadata.{k}",
                    match=MatchValue(value=v),
                )
                for k, v in filters.items()
            ]
            query_filter = Filter(must=conditions)

        results = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            using="dense",
            limit=top_k,
            query_filter=query_filter,
        ).points

        return [
            ScoredChunk(
                chunk=Chunk.model_validate(hit.payload),
                score=hit.score,
                ranking_method="dense",
            )
            for hit in results
        ]

    def search_sparse(
        self,
        text: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        raise NotImplementedError("Sparse search added in Phase 3.")

    def get_by_ids(self, ids: list[str]) -> list[Chunk]:
        points = self._client.retrieve(collection_name=self._collection, ids=ids)
        return [Chunk.model_validate(p.payload) for p in points]

    def delete_by_source(self, source_path: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        source_filter = Filter(
            must=[FieldCondition(key="metadata.source_path", match=MatchValue(value=source_path))]
        )

        offset = None
        ids_to_delete: list[str] = []

        while True:
            result = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=source_filter,
                limit=100,
                offset=offset,
            )
            points, next_offset = result

            ids_to_delete.extend(p.id for p in points)

            if next_offset is None:
                break
            offset = next_offset

        if ids_to_delete:
            from qdrant_client.models import PointIdsList

            self._client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=ids_to_delete),
            )
            logger.info("Deleted %d points for source '%s'.", len(ids_to_delete), source_path)
