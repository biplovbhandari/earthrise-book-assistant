import logging

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.sql import text

from earthrise_rag.db.base import Base

logger = logging.getLogger(__name__)

_SCHEMA = Base.metadata.schema
_SEARCH_PATH = f"{_SCHEMA},public"


def create_db_engine(database_url: str) -> AsyncEngine:
    """Create an async SQLAlchemy engine with connection pooling."""
    return create_async_engine(
        database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        connect_args={
            "timeout": 5,
            "server_settings": {"search_path": _SEARCH_PATH},
        },
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create a session factory bound to the given engine."""
    return async_sessionmaker(engine, expire_on_commit=False)


async def check_db_schema_status(engine: AsyncEngine) -> str:
    """Check database connectivity and schema state.

    Returns one of: 'ready', 'stale_schema', 'no_schema', 'unavailable'.
    """
    try:
        async with engine.connect() as conn:
            has_table = await conn.execute(
                text(
                    "SELECT EXISTS ("
                    "  SELECT 1 FROM information_schema.tables"
                    "  WHERE table_schema = 'public' AND table_name = 'alembic_version'"
                    ")"
                )
            )
            if not has_table.scalar():
                return "no_schema"
            result = await conn.execute(text("SELECT version_num FROM alembic_version"))
            row = result.first()
            if not row:
                return "no_schema"
            db_version = row[0]
            head = _get_alembic_head()
            return "ready" if db_version == head else "stale_schema"
    except Exception:
        return "unavailable"


def _get_alembic_head() -> str | None:
    """Return the packaged Alembic head revision, or None if unavailable."""
    try:
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        cfg = Config("alembic.ini")
        script = ScriptDirectory.from_config(cfg)
        return script.get_current_head()
    except Exception:
        return None
