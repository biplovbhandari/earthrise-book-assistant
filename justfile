# EarthRISE Book Assistant
# Run `just --list` to see all available recipes.

# ------------------ Daily development -----------------------------

# Start backing services and the dev server with hot-reload
dev: services
    uv run uvicorn api.main:app --reload

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

# ------------------ Content ---------------------------------------

# Index book chapters, PDFs, and transcripts into Qdrant
index *args='':
    BOOK_COMMIT_SHA=$(git -C book rev-parse HEAD) uv run python scripts/index_book.py {{ args }}

# Transcribe YouTube lecture videos
transcribe *args='':
    uv run --group indexer python scripts/transcribe.py {{ args }}

# Render the book with the chat widget and copy to _book/
render-book:
    docker compose --profile build run --rm quarto-builder
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
