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
    docker run --rm \
        -v earthrise-book-assistant_book_html:/src \
        -v "$(pwd)/_book":/dst \
        alpine sh -c 'cp -a /src/. /dst/'

# ------------------ Docker services -------------------------------

# Start Qdrant and PostgreSQL in the background
services:
    docker compose up qdrant postgres -d

# Start the full Docker stack (production-like)
up:
    docker compose up -d

# Start the full Docker stack with hot-reload (dev override)
dev-docker:
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Stop all Docker services
down:
    docker compose down

# Rebuild Docker images
build:
    docker compose build app quarto-builder indexer
