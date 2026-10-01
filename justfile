# EarthRISE Book Assistant
# Run `just --list` to see all available recipes.

# ------------------ Daily development -----------------------------

# Start backing services and the dev server with hot-reload
dev: services
    uv run uvicorn api.main:app --reload

# Run the app in production mode (no hot-reload, binds to all interfaces)
serve: services
    uv run uvicorn api.main:app --host 0.0.0.0 --port 8000

# Run lint, type check, and tests
check: lint typecheck test

# Auto-fix formatting
format:
    uv run ruff format .

# ------------------ Quality checks -------------------------------

# Run ruff linter and format check
lint:
    uv run ruff check .
    uv run ruff format --check .

# Run pyright type checker
typecheck:
    uv run pyright

# Run the test suite
test *args='':
    uv run pytest tests/ -v {{ args }}

# ------------------ Database --------------------------------------

# Apply pending Alembic migrations
db-migrate:
    uv run alembic upgrade head

# Create a new auto-generated migration
db-revision msg:
    uv run alembic revision --autogenerate -m "{{ msg }}"

# Dump PostgreSQL to a timestamped compressed file in backups/
db-backup: (_db-backup-prune)
    #!/usr/bin/env bash
    set -euo pipefail
    if ! docker exec earthrise-db pg_isready -U earthrise -d earthrise >/dev/null 2>&1; then
        echo "Error: PostgreSQL is not running" >&2
        exit 1
    fi
    mkdir -p backups
    outfile="backups/earthrise_$(date +%Y%m%d_%H%M%S).sql.gz"
    docker exec earthrise-db pg_dump -U earthrise earthrise | gzip > "$outfile"
    echo "Backup: $outfile ($(du -h "$outfile" | cut -f1))"

# Remove old backups, keeping the 30 most recent
_db-backup-prune:
    #!/usr/bin/env bash
    set -euo pipefail
    [[ -d backups ]] || exit 0
    count=$(ls -1 backups/earthrise_*.sql.gz 2>/dev/null | wc -l)
    if (( count > 30 )); then
        ls -1t backups/earthrise_*.sql.gz | tail -n +"31" | xargs rm -f
        echo "Pruned $((count - 30)) old backup(s)"
    fi

# Restore PostgreSQL from a backup file (replaces current data)
db-restore file:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ ! -f "{{ file }}" ]]; then
        echo "File not found: {{ file }}" >&2
        exit 1
    fi
    echo "WARNING: This will replace all data in the earthrise database."
    read -p "Continue? [y/N] " -n 1 -r; echo
    [[ $REPLY =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 1; }
    docker exec earthrise-db dropdb -U earthrise --force --if-exists earthrise
    docker exec earthrise-db createdb -U earthrise earthrise
    gunzip -c "{{ file }}" | docker exec -i earthrise-db psql -U earthrise -d earthrise --quiet
    echo "Restored from: {{ file }}"
    echo "Run 'just db-migrate' if the backup predates a schema change."

# ------------------ Content ---------------------------------------

# Index book chapters, PDFs, and transcripts into Qdrant (local)
index *args='':
    BOOK_COMMIT_SHA=$(git -C book rev-parse HEAD) uv run python scripts/index_book.py {{ args }}

# Index using Docker indexer (production deployment, no uv needed)
index-prod *args='':
    BOOK_COMMIT_SHA=$(git -C book rev-parse HEAD) docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile build run --rm indexer {{ args }}

# Transcribe YouTube lecture videos
transcribe *args='':
    uv run --group indexer python scripts/transcribe.py {{ args }}

# Render the book with the chat widget and copy to _book/ (local build)
render-book:
    docker compose --profile build run --rm quarto-builder
    mkdir -p _book
    docker compose --profile build run --rm book-copy

# Render the book using GHCR image (production deployment)
render-book-prod:
    docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile build run --rm quarto-builder
    mkdir -p _book
    docker compose --profile build run --rm book-copy

# ------------------ Docker services -------------------------------

# Start Qdrant and PostgreSQL in the background
services:
    docker compose up qdrant postgres -d
    @echo "Waiting for PostgreSQL..."
    @until docker exec earthrise-db psql -U earthrise -d earthrise -c "SELECT 1" >/dev/null 2>&1; do sleep 1; done
    @echo "PostgreSQL ready."

# Start the full Docker stack (local build)
up:
    docker compose up -d

# Start the full Docker stack using GHCR images
up-prod:
    docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Start the full Docker stack with hot-reload (dev override)
dev-docker:
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Stop all Docker services
down:
    docker compose down

# Reset the database only (keeps Qdrant index intact)
db-reset:
    docker compose stop postgres
    rm -rf .data/postgres
    docker compose up postgres -d
    @echo "Waiting for PostgreSQL..."
    @until docker exec earthrise-db psql -U earthrise -d earthrise -c "SELECT 1" >/dev/null 2>&1; do sleep 1; done
    @echo "PostgreSQL ready."
    uv run alembic upgrade head

# Stop services, remove volumes and data for a fresh start
clean:
    docker compose down -v --remove-orphans
    -docker rm -f $(docker ps -aq --filter "label=com.docker.compose.project=earthrise-book-assistant") 2>/dev/null
    -docker volume ls -q --filter "name=earthrise-book-assistant" | xargs -r docker volume rm 2>/dev/null
    rm -rf .data/qdrant .data/postgres _book

# Rebuild Docker images
build:
    docker compose build app quarto-builder indexer
