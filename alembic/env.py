import asyncio
from logging.config import fileConfig

from sqlalchemy import pool, text
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from earthrise_rag.config import get_settings
from earthrise_rag.db import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
_SCHEMA = target_metadata.schema
_SEARCH_PATH = f"{_SCHEMA},public"

_SHARED_CONFIGURE = {
    "target_metadata": target_metadata,
    "compare_type": True,
    "include_schemas": True,
    "version_table_schema": "public",
}


def _get_url() -> str:
    """Resolve database URL: -x override first, then alembic.ini, then Settings."""
    cmd_url = context.get_x_argument(as_dictionary=True).get("sqlalchemy.url", "")
    if cmd_url:
        return cmd_url
    ini_url = config.get_main_option("sqlalchemy.url", "")
    if ini_url:
        return ini_url
    return get_settings().database_url.get_secret_value()


def run_migrations_offline() -> None:
    """Generate SQL without connecting to the database."""
    context.configure(
        url=_get_url(),
        literal_binds=True,
        dialect_name="postgresql",
        **_SHARED_CONFIGURE,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Run migrations within a sync connection callback."""
    context.configure(connection=connection, **_SHARED_CONFIGURE)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations."""
    connectable = create_async_engine(
        _get_url(),
        poolclass=pool.NullPool,
        connect_args={
            "timeout": 10,
            "server_settings": {"search_path": _SEARCH_PATH},
        },
    )
    try:
        async with connectable.connect() as connection:
            await connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {_SCHEMA}"))
            await connection.commit()
            await connection.run_sync(do_run_migrations)
    finally:
        await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
