from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from earthrise_rag.models.chunk import Chunk
from earthrise_rag.models.scored_chunk import ScoredChunk

if TYPE_CHECKING:
    from earthrise_rag.interfaces import SparseEmbedder

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100
_TOP_LEVEL_FIELDS = {"source_type", "content_type", "chunk_type"}


class QdrantStore:
    """Qdrant-backed vector store with dense and sparse search."""

    def __init__(
        self,
        url: str,
        collection_name: str,
        dense_dim: int,
        create_if_missing: bool = True,
        sparse_embedder: SparseEmbedder | None = None,
    ) -> None:
        """Connect to Qdrant and validate (or create) the collection.

        Imports are constructor-scoped so the qdrant-client package is
        loaded only when a QdrantStore is actually instantiated.
        """
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance,
            Modifier,
            SparseIndexParams,
            SparseVectorParams,
            VectorParams,
        )

        self._client = QdrantClient(url=url)
        self._collection = collection_name
        self._dense_dim = dense_dim
        self._sparse_embedder = sparse_embedder
        self._has_sparse = False
        self._TOP_LEVEL_FIELDS = _TOP_LEVEL_FIELDS

        self._ensure_collection(
            Distance,
            VectorParams,
            SparseVectorParams,
            SparseIndexParams,
            Modifier,
            create_if_missing,
        )

    def _ensure_collection(
        self,
        Distance,
        VectorParams,
        SparseVectorParams,
        SparseIndexParams,
        Modifier,
        create_if_missing: bool,
    ) -> None:
        """Validate or create the collection and set ``_has_sparse``.

        Behavior depends on collection state and caller context:

        - **Exists with sparse** -- validate dense config, set ``_has_sparse = True``.
        - **Exists without sparse, indexer** (``create_if_missing=True``) -- raise so the
          operator runs ``--recreate-collection`` (Qdrant cannot add vector types to
          an existing collection).
        - **Exists without sparse, app** (``create_if_missing=False``) -- warn and
          degrade; dense search still works, sparse returns empty.
        - **Missing, indexer** -- create with both dense + sparse config.
        - **Missing, app** -- log not-ready; ``_has_sparse`` stays ``False``.
        """
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
            else:
                raise ValueError(
                    f"Collection '{self._collection}' exists but has no 'dense' named vector."
                )

            sparse_config = info.config.params.sparse_vectors
            has_sparse = sparse_config is not None and "sparse" in sparse_config

            if not has_sparse and create_if_missing:
                raise ValueError(
                    f"Collection '{self._collection}' has no 'sparse' named vector. "
                    f"Run the indexer with --recreate-collection to add sparse support."
                )
            elif not has_sparse:
                logger.warning(
                    "Collection '%s' has no 'sparse' named vector. "
                    "Hybrid search will not work until re-indexed with --recreate-collection.",
                    self._collection,
                )
                self._has_sparse = False
            else:
                self._has_sparse = True

            logger.info("Collection '%s' exists with matching config.", self._collection)

        elif create_if_missing:
            use_idf = self._sparse_embedder is not None and self._sparse_embedder.requires_idf
            sparse_params = SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=Modifier.IDF if use_idf else None,
            )
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    "dense": VectorParams(size=self._dense_dim, distance=Distance.COSINE)
                },
                sparse_vectors_config={"sparse": sparse_params},
            )
            self._has_sparse = True
            logger.info(
                "Created collection '%s' (dense_dim=%d, sparse=True).",
                self._collection,
                self._dense_dim,
            )
        else:
            logger.info(
                "Collection '%s' does not exist yet; query path will return not-ready.",
                self._collection,
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Store chunks with dense vectors and optional sparse vectors.

        Sparse embeddings are computed upfront for all chunks before the
        batch write loop so that a SPLADE failure never leaves Qdrant in
        a partially-written state within this call.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must have equal length"
            )

        from qdrant_client.models import PointStruct, SparseVector

        if self._sparse_embedder is not None and self._has_sparse:
            all_sparse = self._sparse_embedder.embed([c.content for c in chunks])
            if len(all_sparse) != len(chunks):
                raise ValueError(
                    f"Sparse embedding returned {len(all_sparse)} vectors for {len(chunks)} chunks"
                )
        else:
            all_sparse = [None] * len(chunks)

        for i in range(0, len(chunks), _BATCH_SIZE):
            batch_chunks = chunks[i : i + _BATCH_SIZE]
            batch_vectors = vectors[i : i + _BATCH_SIZE]
            batch_sparse = all_sparse[i : i + _BATCH_SIZE]

            batch_points = []
            for chunk, dense_vec, sparse_emb in zip(batch_chunks, batch_vectors, batch_sparse):
                vec_dict: dict = {"dense": dense_vec}
                if sparse_emb is not None:
                    vec_dict["sparse"] = SparseVector(
                        indices=list(sparse_emb.indices),
                        values=list(sparse_emb.values),
                    )
                batch_points.append(
                    PointStruct(id=chunk.id, vector=vec_dict, payload=chunk.model_dump())
                )

            self._client.upsert(collection_name=self._collection, points=batch_points)

    def _build_filter(self, filters: dict[str, Any] | None):
        """Chunk payload stores source_type/content_type/chunk_type at the top
        level, everything else under metadata.*. Map filter keys accordingly."""
        if not filters:
            return None

        from qdrant_client.models import Condition, FieldCondition, Filter, MatchValue

        conditions: list[Condition] = [
            FieldCondition(
                key=k if k in self._TOP_LEVEL_FIELDS else f"metadata.{k}",
                match=MatchValue(value=v),
            )
            for k, v in filters.items()
        ]
        return Filter(must=conditions)

    def search_dense(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Query the 'dense' named vector index via cosine similarity."""
        results = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            using="dense",
            limit=top_k,
            query_filter=self._build_filter(filters),
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
        """Returns [] instead of raising when sparse is unavailable (no
        embedder, old collection, or empty encoding) so HybridStrategy
        can fall back to dense-only without try/except."""
        if not self._sparse_embedder or not self._has_sparse:
            return []

        from qdrant_client.models import SparseVector

        sparse_embeddings = self._sparse_embedder.query_embed(text)

        if not sparse_embeddings:
            return []

        sparse_vec = sparse_embeddings[0]

        if len(sparse_vec.indices) == 0:
            return []

        query_vector = SparseVector(
            indices=list(sparse_vec.indices),
            values=list(sparse_vec.values),
        )

        results = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            using="sparse",
            limit=top_k,
            query_filter=self._build_filter(filters),
        ).points

        return [
            ScoredChunk(
                chunk=Chunk.model_validate(hit.payload),
                score=hit.score,
                ranking_method="sparse",
            )
            for hit in results
        ]

    def get_by_ids(self, ids: list[str]) -> list[Chunk]:
        """Retrieve chunks by their Qdrant point IDs."""
        points = self._client.retrieve(collection_name=self._collection, ids=ids)
        return [Chunk.model_validate(p.payload) for p in points]

    def count(self) -> int:
        """Approximate point count; returns 0 if the collection is missing."""
        try:
            result = self._client.count(collection_name=self._collection, exact=False)
            return result.count
        except Exception:
            return 0

    def delete_by_source(self, source_path: str) -> None:
        """Scroll all points matching metadata.source_path and delete them."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        source_filter = Filter(
            must=[FieldCondition(key="metadata.source_path", match=MatchValue(value=source_path))]
        )

        offset = None
        ids_to_delete: list = []

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
