"""Prometheus metrics for RAG pipeline observations."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

llm_generation_seconds = Histogram(
    "llm_generation_seconds",
    "LLM generation duration in seconds",
)

retrieval_seconds = Histogram(
    "retrieval_seconds",
    "Retrieval duration in seconds",
)

retrieval_chunk_count = Histogram(
    "retrieval_chunk_count",
    "Number of chunks returned per retrieval",
    buckets=[0, 1, 2, 3, 5, 8, 10, 15, 20],
)

search_request_count = Counter(
    "search_request_count",
    "Total search requests",
)
