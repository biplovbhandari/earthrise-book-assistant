#!/bin/sh
set -e

if [ -n "$DATABASE_URL" ]; then
    MAX_RETRIES=5
    RETRY_DELAY=3
    attempt=0
    while [ $attempt -lt $MAX_RETRIES ]; do
        if uv run --frozen alembic upgrade head; then
            echo "Migrations applied successfully"
            break
        fi
        attempt=$((attempt + 1))
        if [ $attempt -lt $MAX_RETRIES ]; then
            echo "Migration attempt $attempt/$MAX_RETRIES failed; retrying in ${RETRY_DELAY}s..."
            sleep $RETRY_DELAY
        fi
    done
    if [ $attempt -eq $MAX_RETRIES ]; then
        echo "WARNING: Migrations failed after $MAX_RETRIES attempts; starting app anyway"
    fi
fi

exec "$@"
