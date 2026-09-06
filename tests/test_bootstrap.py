"""Unit tests for the pure helper functions in earthrise_rag.db.bootstrap.

ensure_active_deployment itself opens a real PostgreSQL session and is out
of scope here; only the database-free config-snapshot and prompt-hashing
logic is covered.
"""

from pydantic import SecretStr

from earthrise_rag.config import Settings
from earthrise_rag.db.bootstrap import _build_config_snapshot, _read_prompt


class TestBuildConfigSnapshot:
    def test_includes_allowlisted_fields(self):
        """Config snapshot includes only the allowlisted fields."""
        settings = Settings(
            embedding_model_name="test-model",
            retrieval_strategy="hybrid",
            llm_model="gpt-4",
            reranker_provider="noop",
            sparse_model_name="splade",
            database_url=SecretStr("postgresql://secret"),
            llm_api_key=SecretStr("sk-secret"),
        )
        snapshot = _build_config_snapshot(settings)

        assert snapshot["embedding_model_name"] == "test-model"
        assert snapshot["retrieval_strategy"] == "hybrid"
        assert snapshot["llm_model"] == "gpt-4"
        assert snapshot["reranker_provider"] == "noop"
        assert snapshot["sparse_model_name"] == "splade"

    def test_excludes_secrets(self):
        """Config snapshot must not contain database_url or llm_api_key."""
        settings = Settings(
            database_url=SecretStr("postgresql://secret"),
            llm_api_key=SecretStr("sk-secret"),
        )
        snapshot = _build_config_snapshot(settings)

        assert "database_url" not in snapshot
        assert "llm_api_key" not in snapshot
        for value in snapshot.values():
            assert "secret" not in str(value).lower()


class TestReadPrompt:
    def test_returns_hash_and_content(self):
        """_read_prompt returns a (hash, content) tuple with non-empty values."""
        prompt_hash, content = _read_prompt()

        assert len(prompt_hash) == 64  # SHA-256 hex digest
        assert len(content) > 0
        assert content == content.strip()
