"""Tests for database engine factory and session management."""

import pytest

from earthrise_rag.db.engine import create_db_engine, create_session_factory

_DUMMY_URL = "postgresql+asyncpg://user:pass@localhost/db"


@pytest.fixture
def engine():
    """Create and dispose a test engine with a dummy URL."""
    e = create_db_engine(_DUMMY_URL)
    yield e
    e.sync_engine.dispose()


def test_create_engine_returns_async_engine(engine):
    """Engine factory returns an AsyncEngine with expected pool config."""
    assert engine.url.drivername == "postgresql+asyncpg"
    assert engine.pool.size() == 5


def test_create_session_factory_returns_maker(engine):
    """Session factory is bound to the engine and uses expire_on_commit=False."""
    factory = create_session_factory(engine)
    assert factory.kw.get("expire_on_commit") is False


def test_engine_pool_pre_ping_enabled(engine):
    """pool_pre_ping is enabled for runtime recovery after DB restarts."""
    assert engine.pool._pre_ping is True
