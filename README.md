# earthrise-book-assistant

Book viewer and retrieval-augmented generation (RAG) assistant for the [EarthRISE Applied AI and Deep Learning Book](https://nasa-earthrise.github.io/EarthRISE-Applied-Artificial-Intelligence-and-Deep-Learning-Book/).
Serves Quarto-rendered chapters alongside a search and generation API.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (required for both paths -- runs Qdrant, or the full stack)
- [Ollama](https://ollama.com/) (for `/ask` and `/chat` generation -- pull a model with `ollama pull qwen3:8b`)
- [Python 3.12+](https://www.python.org/downloads/) (local setup only -- not needed for Docker deployment)
- [uv](https://docs.astral.sh/uv/) (local setup only)
- [Node.js](https://nodejs.org/) (optional -- JS/CSS linting only)
- [ffmpeg](https://ffmpeg.org/download.html) (local transcription only -- Docker image includes it)

## Getting Started

```bash
git clone --recurse-submodules <repo-url>
cd earthrise-book-assistant
cp .env.example .env
```

Edit `.env` -- see `.env.example` for all available settings.

## Local Setup (uv)

Runs the app on your machine with uv.
Qdrant runs in Docker (required).
PostgreSQL is optional -- only needed for interaction recording and analytics.
You can use Docker Compose for PostgreSQL or a locally installed PostgreSQL with the pgvector extension.

```bash
# Install dependencies
uv sync --group dev --group indexer

# Configure .env (defaults work for local dev)
cp .env.example .env
# Edit .env if needed -- key settings:
#   LLM_MODEL=qwen3:8b               # must be set
#   RERANKER_PROVIDER=noop            # or local_cross_encoder
#   DATABASE_URL=                     # empty = DB disabled; set for interaction recording

# Start Qdrant (required)
docker compose up qdrant -d

# PostgreSQL (optional -- choose one):
docker compose up postgres -d          # option A: Docker (recommended)
# option B: use your own PostgreSQL with pgvector installed

# Apply database migrations (only if PostgreSQL is running)
DATABASE_URL=postgresql+asyncpg://earthrise:earthrise@localhost:5432/earthrise \
  uv run alembic upgrade head

# Transcribe YouTube lectures (optional -- skip if transcripts are already committed)
uv run --group indexer python scripts/transcribe.py

# Index the book (chapters + companion PDFs + transcripts if available)
BOOK_COMMIT_SHA=$(git -C book rev-parse HEAD) \
  uv run python scripts/index_book.py

# Start the app (requires Ollama running for /ask and /chat)
uv run uvicorn api.main:app --reload
```

Try it:

```bash
# Health check (shows retrieval, generation, chat, and database readiness)
curl -s localhost:8000/health | python3 -m json.tool

# Search (retrieval only)
curl -s -X POST localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"question": "What is U-Net?"}' | python3 -m json.tool

# Ask (generates an answer with citations -- requires Ollama)
curl -s -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "What is U-Net?"}' | python3 -m json.tool

# Chat (streaming SSE -- requires Ollama)
curl -N -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question": "What is semantic segmentation?"}'

# Chat with follow-up history
curl -N -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question": "Tell me more about that", "history": [{"role": "user", "content": "What is U-Net?"}, {"role": "assistant", "content": "A CNN architecture for segmentation."}]}'
```

If the index is empty or Qdrant is unreachable, `/search`, `/ask`, and `/chat` return `503`.
If Ollama is not running, `/search` still works but `/ask` and `/chat` return `503`.

### Render the Book with Chat Widget

The chat widget is injected into every book page during rendering.
For local dev, you need the rendered book in `_book/`.

**Option A: Docker render, copy to local** (recommended if you have a built quarto-builder image):

```bash
docker compose --profile build run --rm quarto-builder
docker run --rm \
  -v earthrise-book-assistant_book_html:/src \
  -v "$(pwd)/_book":/dst \
  alpine sh -c 'cp -a /src/. /dst/'
```

**Option B: Local Quarto render** (requires [Quarto CLI](https://quarto.org/docs/get-started/)):

```bash
rm -rf /tmp/book_render
cp -r book /tmp/book_render && rm -rf /tmp/book_render/.git
cp widget/_quarto-chat.yml /tmp/book_render/_quarto-chat.yml
mkdir -p /tmp/book_render/_includes
cp widget/chat.html /tmp/book_render/_includes/chat.html
printf '<link rel="stylesheet" href="/_widget/chat.css">\n' > /tmp/book_render/_includes/chat-head.html
printf '<script src="/_widget/chat.js"></script>\n' > /tmp/book_render/_includes/chat-foot.html
cd /tmp/book_render && quarto render --profile chat
rm -rf _book/* && cp -a /tmp/book_render/_book/. _book/
mkdir -p _book/_widget
cp widget/chat.css _book/_widget/ && cp widget/chat.js _book/_widget/
```

After rendering, open http://localhost:8000/ to see the book with the chat FAB in the bottom-right corner.

## Index Book Content

Indexes book chapters, companion PDFs, and video transcripts.
Transcripts are committed to the repo, so cloning gives you everything -- no need to run `transcribe.py` first.
The Qdrant `qdrant_data` volume persists across restarts -- you only need to re-index if the volume is removed or content changes.

```bash
# Local
BOOK_COMMIT_SHA=$(git -C book rev-parse HEAD) \
  uv run python scripts/index_book.py

# Docker
BOOK_COMMIT_SHA=$(git -C book rev-parse HEAD) \
  docker compose --profile build run --rm indexer

# Fresh index (delete and recreate the Qdrant collection first)
# Add --recreate-collection to either command above, e.g.:
BOOK_COMMIT_SHA=$(git -C book rev-parse HEAD) \
  uv run python scripts/index_book.py --recreate-collection
```

Check the [Qdrant dashboard](http://localhost:6333/dashboard) for indexed chunks.
Logs written to `logs/`.

## Transcribe YouTube Lectures (Optional)

Downloads audio from the book's YouTube playlist and transcribes with Whisper.
Transcripts are saved to `data/transcripts/` and committed to the repo.
Transcription is optional -- if you want video content searchable, transcribe before indexing.

```bash
# Docker
docker compose --profile build run --rm indexer \
  uv run python scripts/transcribe.py

# Docker -- single video
docker compose --profile build run --rm indexer \
  uv run python scripts/transcribe.py --video-id <VIDEO_ID>

# Local
uv run --group indexer python scripts/transcribe.py

# Local -- single video
uv run --group indexer python scripts/transcribe.py --video-id <VIDEO_ID>

# Re-transcribe all (add --force to either Docker or local command)

# Download audio only (for GPU transcription on Colab)
uv run --group indexer python scripts/transcribe.py --download-only
# Then zip data/audio/, upload to Colab, and run notebooks/transcribe_gpu.ipynb
```

For GPU-accelerated transcription, use `notebooks/transcribe_gpu.ipynb` in Google Colab.

After transcribing, update `data/video_chapter_map.yml` to map video IDs to chapters:

```yaml
videos:
  dQw4w9WgXcQ:
    chapter: "03_Semantic_Segmentation"
    lesson: "01__Crop_Mapping"
```

Video IDs are the JSON filenames in `data/transcripts/` (e.g. `dQw4w9WgXcQ.json`).
Chapter and lesson values match directory names under `book/`.
Then re-run the indexer to include transcripts.

Requires `ffmpeg` installed.
See [ffmpeg.org/download.html](https://ffmpeg.org/download.html) for installation instructions.

## Docker Deployment

Complete Docker path -- no local Python required.
Complete the [Getting Started](#getting-started) steps first (clone + .env).
Docker Compose overrides `QDRANT_URL`, `LLM_BASE_URL`, and `DATABASE_URL` automatically.
Database migrations run automatically on app startup via the entrypoint script.

```bash
# Setup
cp .env.example .env
# Edit .env -- set LLM_MODEL for /ask and /chat generation.

# Build and render
docker compose build app quarto-builder indexer
docker compose --profile build run --rm quarto-builder     # render the book

# Transcribe YouTube lectures (optional -- skip if transcripts are already committed)
docker compose --profile build run --rm indexer \
  uv run python scripts/transcribe.py

# Index content (chapters + PDFs + transcripts)
BOOK_COMMIT_SHA=$(git -C book rev-parse HEAD) \
  docker compose --profile build run --rm indexer \
  uv run python scripts/index_book.py --recreate-collection

# Start the app
docker compose up -d
```

Try it:

```bash
# Health check
curl -s localhost:8000/health | python3 -m json.tool

# Search
curl -s -X POST localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"question": "What is U-Net?"}' | python3 -m json.tool

# Ask (requires Ollama on the host or a remote LLM)
curl -s -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "What is U-Net?"}' | python3 -m json.tool

# Chat (streaming SSE -- requires Ollama)
curl -N -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question": "What is semantic segmentation?"}'
```

- Book with chat widget: http://localhost:8000/
- Qdrant dashboard: http://localhost:6333/dashboard

### Re-render book

```bash
docker compose --profile build run --rm quarto-builder
```

Code changes require `docker compose build app`.
For hot reload:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

To stop: `docker compose down`

## Database Migrations

Schema is managed by Alembic.
SQLAlchemy ORM models are the source of truth; Alembic generates migration scripts from model changes.

```bash
# Apply pending migrations
DATABASE_URL=postgresql+asyncpg://earthrise:earthrise@localhost:5432/earthrise \
  uv run alembic upgrade head

# Generate a new migration after changing ORM models
DATABASE_URL=postgresql+asyncpg://earthrise:earthrise@localhost:5432/earthrise \
  uv run alembic revision --autogenerate -m "describe the change"

# Generate standalone SQL (for review or manual execution)
uv run alembic -x sqlalchemy.url=postgresql+asyncpg://x:x@localhost/x \
  upgrade head --sql > dev/artifacts/db/001_initial_schema.sql
```

Docker Compose applies migrations automatically on app startup via the entrypoint script.

## Testing and Linting

```bash
uv sync --group dev --group indexer
uv run pytest -v -m "not integration"                      # unit tests (no DB needed)
uv run ruff check . && uv run ruff format --check .        # Python lint
uv run pyright                                             # type check

# Integration tests (requires PostgreSQL running -- tests against real DB)
docker compose up postgres -d
TEST_DATABASE_URL=postgresql+asyncpg://earthrise:earthrise@localhost:5432/earthrise \
  uv run pytest -m integration -v                          # migration tests

# JS/CSS linting (optional -- requires Node.js)
npm install                                                # first time only
npx eslint widget/chat.js                                  # JS lint
npx stylelint widget/chat.css                              # CSS lint
```

## Dependency Management

Dependencies are managed with [uv](https://docs.astral.sh/uv/).
`uv.lock` is committed to the repo for reproducible installs.

```bash
uv sync --group dev --group indexer    # install all dependencies
uv lock                                # regenerate uv.lock after editing pyproject.toml
uv sync                                # install after regenerating lock
```

## Project Structure

```
earthrise-book-assistant/
├── src/earthrise_rag/           # Python package (RAG logic)
│   ├── config.py                # Pydantic BaseSettings, env-driven
│   ├── interfaces.py            # Shared protocols (Embedder, SparseEmbedder, VectorStore, ...)
│   ├── models/                  # Chunk, ScoredChunk, Document, Answer, Citation, IndexResult
│   ├── db/                      # SQLAlchemy ORM models, engine factory, session management
│   ├── indexing/                # Parsers, chunkers, embedder, sparse embedder, vector store, pipeline
│   ├── retrieval/               # DenseStrategy, HybridStrategy (RRF), NoOpReranker, LocalCrossEncoderReranker
│   ├── generation/              # LLM client, context builder, system prompt
│   ├── citations/               # Citation builder
│   └── query/                   # QueryPipeline (search + ask)
├── api/                         # FastAPI app
│   ├── main.py                  # /health, lifespan (DB engine), routers, static book mount
│   ├── dependencies.py          # Adapter wiring (factories, DB session providers)
│   └── routes/                  # /search, /ask, /chat endpoints + readiness helpers
├── scripts/
│   ├── index_book.py            # CLI: index book + PDFs + transcripts into Qdrant
│   ├── transcribe.py            # CLI: download + transcribe YouTube lectures
│   └── run_query.py             # CLI: search the index
├── data/
│   ├── video_chapter_map.yml    # Maps video IDs to book chapters
│   └── transcripts/             # Whisper-generated JSON transcripts (committed)
├── notebooks/                   # Colab notebooks (GPU transcription)
├── widget/                      # Chat widget (injected into book pages by Quarto)
│   ├── chat.css                 # Widget styles (pure CSS, lintable)
│   ├── chat.js                  # Widget logic (pure JS, lintable)
│   ├── chat.html                # Widget HTML structure
│   └── _quarto-chat.yml         # Quarto profile overlay
├── alembic/                     # Database migrations (Alembic)
├── infra/docker/                # Dockerfiles + entrypoint script
├── tests/
├── system-design/               # Architecture docs
├── book/                        # Git submodule (book source)
├── docker-compose.yml
├── .env.example                 # Config template (all available settings)
└── pyproject.toml
```

## License

Apache License 2.0 -- see [LICENSE](LICENSE).
