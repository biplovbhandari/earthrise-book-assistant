"""Integration tests for Alembic migrations against a real PostgreSQL database.

Requires TEST_DATABASE_URL env var pointing to a PostgreSQL server.
The test creates a uniquely named temporary database, runs all assertions,
and drops it in finally - never downgrades a caller-owned database.

Run with: uv run pytest -m integration tests/test_db_migration.py -v
"""

import asyncio
import os
import subprocess
import uuid
from urllib.parse import urlparse

import asyncpg
import pytest

from earthrise_rag.db.base import Base

pytestmark = pytest.mark.integration

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
_SCHEMA = Base.metadata.schema

_TABLES_SQL = (
    f"SELECT tablename FROM pg_tables"
    f" WHERE schemaname = '{_SCHEMA}' AND tablename != 'alembic_version'"
)
_VIEWS_SQL = f"SELECT viewname FROM pg_views WHERE schemaname = '{_SCHEMA}'"


def _parse_server_url(url: str) -> tuple[str, str, str, int]:
    """Extract user, password, host, port from a PostgreSQL URL."""
    parsed = urlparse(url.replace("postgresql+asyncpg://", "postgresql://"))
    return (
        parsed.username or "earthrise",
        parsed.password or "earthrise",
        parsed.hostname or "localhost",
        parsed.port or 5432,
    )


async def _admin_execute(sql: str) -> None:
    """Run a DDL statement against the postgres maintenance database."""
    user, password, host, port = _parse_server_url(TEST_DATABASE_URL)
    conn = await asyncpg.connect(
        user=user, password=password, host=host, port=port, database="postgres"
    )
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


def _query(db_url: str, sql: str) -> list:
    """Synchronous wrapper: run a query against a database and return rows."""

    async def _run():
        user, password, host, port = _parse_server_url(db_url)
        db_name = db_url.rsplit("/", 1)[-1]
        conn = await asyncpg.connect(
            user=user, password=password, host=host, port=port, database=db_name
        )
        try:
            return await conn.fetch(sql)
        finally:
            await conn.close()

    return asyncio.run(_run())


def _alembic_cmd(db_url: str, command: str, revision: str = "head") -> None:
    """Run alembic as a subprocess to avoid nested event loop issues."""
    env = {**os.environ, "DATABASE_URL": ""}
    result = subprocess.run(
        ["uv", "run", "--frozen", "alembic", "-x", f"sqlalchemy.url={db_url}", command, revision],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"alembic {command} failed:\n{result.stderr}")


@pytest.fixture(scope="module")
def temp_db():
    """Create a temporary database, yield its URL, drop it when done."""
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set")

    db_name = f"test_earthrise_{uuid.uuid4().hex[:8]}"
    user, password, host, port = _parse_server_url(TEST_DATABASE_URL)
    db_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"

    asyncio.run(_admin_execute(f'CREATE DATABASE "{db_name}"'))
    try:
        yield db_url
    finally:
        asyncio.run(_admin_execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))


def test_upgrade_creates_all_tables_and_views(temp_db):
    """alembic upgrade head creates all 15 tables and 7 views."""
    _alembic_cmd(temp_db, "upgrade", "head")

    tables = _query(temp_db, _TABLES_SQL)
    assert len(tables) == 15

    views = _query(temp_db, _VIEWS_SQL)
    view_names = {r["viewname"] for r in views}
    assert view_names == {
        "v_interaction_summary",
        "v_citation_heatmap",
        "v_retrieval_gaps",
        "v_deployment_metrics",
        "v_daily_stats",
        "v_conversation_summary",
        "v_eval_run_summary",
    }


def test_downgrade_and_reupgrade(temp_db):
    """downgrade base removes everything, upgrade head recreates it."""
    _alembic_cmd(temp_db, "downgrade", "base")

    tables = _query(temp_db, _TABLES_SQL)
    assert len(tables) == 0

    _alembic_cmd(temp_db, "upgrade", "head")

    tables = _query(temp_db, _TABLES_SQL)
    assert len(tables) == 15


def test_upgrade_is_idempotent(temp_db):
    """Running upgrade head twice succeeds (no-op second time)."""
    _alembic_cmd(temp_db, "upgrade", "head")
    _alembic_cmd(temp_db, "upgrade", "head")
