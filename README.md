# earthrise-book-assistant

Book viewer and retrieval-augmented generation (RAG) assistant for the [EarthRISE Applied AI and Deep Learning Book](https://nasa-earthrise.github.io/EarthRISE-Applied-Artificial-Intelligence-and-Deep-Learning-Book/).
Serves Quarto-rendered chapters alongside a search and generation API.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) - runs Qdrant and PostgreSQL, and can run the full application stack.
- [Ollama](https://ollama.com/) - powers `/ask` and `/chat` generation.
  Pull a model with `ollama pull qwen3:8b`.
- [Python 3.12+](https://www.python.org/downloads/) - runs the app and CLI scripts directly on your machine.
  Not required for the Docker-only deployment path.
- [uv](https://docs.astral.sh/uv/) - manages the Python environment and dependencies.
  Not required for the Docker-only deployment path.
- [just](https://just.systems/) - task runner used for every command in this README.
  Works on macOS, Linux, and Windows.
- [Node.js](https://nodejs.org/) (optional) - only needed to lint the chat widget's JS and CSS.
- [ffmpeg](https://ffmpeg.org/download.html) (optional) - only needed for local transcription.
  The Docker image already includes it.

## Getting Started

```bash
git clone --recurse-submodules <repo-url>
cd earthrise-book-assistant
cp .env.example .env
```

Edit `.env` to configure your setup.
See `.env.example` for all available settings.

For local development (not needed for the Docker-only deployment path):

```bash
uv sync --group dev --group indexer
```

Once installed, the common workflows are:

```bash
just dev      # start Qdrant + PostgreSQL, then the dev server with hot-reload
just index    # index book content into Qdrant
just check    # lint, type check, and run tests
```

Run `just --list` to see every available recipe.
The justfile contains the underlying shell commands for each recipe.

## Development

`just dev` starts Qdrant and PostgreSQL in Docker, then runs the API locally with `uvicorn --reload`.
Python code changes restart the server automatically, so this is the fastest loop for local development.

Try the API:

```bash
# Health check (shows retrieval, generation, chat, and database readiness)
curl -s localhost:8000/health | python3 -m json.tool

# Search (retrieval only)
curl -s -X POST localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"question": "What is U-Net?"}' | python3 -m json.tool

# Ask (generates an answer with citations - requires Ollama)
curl -s -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "What is U-Net?"}' | python3 -m json.tool

# Chat (streaming SSE - requires Ollama)
curl -N -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question": "What is semantic segmentation?"}'
```

If the index is empty or Qdrant is unreachable, `/search`, `/ask`, and `/chat` return `503`.
If Ollama is not running, `/search` still works but `/ask` and `/chat` return `503`.

### Render the Book

The chat widget is injected into every book page during rendering, so you need a rendered copy in `_book/` to see it locally.
Run `just render-book` to render via Docker and copy the output into `_book/`.
Open http://localhost:8000/ to see the book with the chat FAB in the bottom-right corner.

## Content Management

### Indexing

`just index` indexes book chapters, companion PDFs, and video transcripts into Qdrant.
Transcripts are committed to the repo, so a fresh clone already has everything needed.
You do not need to run the transcriber first.
Qdrant data persists across restarts in `.data/qdrant`, so you only need to re-index when the data is removed or the content changes.
Use `just index --recreate-collection` to delete and rebuild the collection from scratch.

Check the [Qdrant dashboard - http://localhost:6333/dashboard](http://localhost:6333/dashboard) to see indexed chunks.
Logs are written to `logs/`.

### Transcription

`just transcribe` downloads audio from the book's YouTube playlist and transcribes it with Whisper.
Transcripts are saved to `data/transcripts/` and committed to the repo.
Transcription is optional and only needed if you want video content to be searchable.
It requires `ffmpeg`.
See the [ffmpeg download page - https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) for installation instructions.
Pass flags to target a specific video or force re-transcription.
Run `just transcribe --help` for the full list.

After transcribing, map each video to its book chapter in `data/video_chapter_map.yml`:

```yaml
videos:
  dQw4w9WgXcQ:
    chapter: "03_Semantic_Segmentation"
    lesson: "01__Crop_Mapping"
```

Video IDs are the JSON filenames in `data/transcripts/` (for example, `dQw4w9WgXcQ.json`).
Chapter and lesson values match directory names under `book/`.
Re-run `just index` afterward to include the new transcripts.

## Database

Schema is managed by Alembic.
SQLAlchemy ORM models are the source of truth, and Alembic generates migration scripts from model changes.
PostgreSQL itself is optional.
The RAG endpoints work without it, and it is only used for interaction recording and analytics.

`just db-migrate` applies pending migrations.
`just db-revision "describe the change"` generates a new migration after you edit the ORM models.
`just db-reset` wipes and recreates the database without touching Qdrant (no re-indexing needed).

Docker Compose applies migrations automatically on app startup, through the entrypoint script.
`just up` and `just dev-docker` do not need a manual migration step as a result.
Run `just db-migrate` yourself for the local `just dev` workflow.

## Testing

`just check` runs everything: lint, type check, and tests.
Run the pieces individually with `just lint`, `just format`, `just typecheck`, and `just test`.
`just test` accepts arguments, for example `just test -k "test_chat"` to run a single test.

Integration tests exercise a real PostgreSQL database, so they are not part of `just check`:

```bash
docker compose up postgres -d
TEST_DATABASE_URL=postgresql+asyncpg://earthrise:earthrise@localhost:5432/earthrise \
  uv run pytest -m integration -v
```

The chat widget's JS and CSS sit outside the Python toolchain, so they are linted separately:

```bash
npm install
npx eslint widget/chat.js
npx stylelint widget/chat.css
```

## Deployment

See [docs/deployment/guide.md](docs/deployment/guide.md) for the full deployment guide with platform-specific instructions.

The deployment differs by platform because Docker on macOS cannot access the Metal GPU:

- **macOS**: `just services` (Docker: Qdrant + Postgres) + `just serve` (native: app with Metal GPU)
- **Linux**: `just up-prod` (full Docker stack with CUDA GPU)

Both platforms use Ollama natively for LLM generation.
See [system-design/deployment-models.md](system-design/deployment-models.md) for model recommendations across different memory configurations.

Use `just down` to stop all services.
Use `just clean` to stop services and wipe all data (volumes, Qdrant, Postgres, rendered book) for a fresh start.

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
│   ├── middleware.py            # Per-IP rate limiting
│   ├── recording.py             # Chat interaction recording (BackgroundTask)
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
├── docs/deployment/             # Deployment guide
├── infra/docker/                # Dockerfiles + entrypoint script
├── infra/launchd/               # macOS launchd service config
├── system-design/               # Architecture and deployment docs
├── tests/
├── book/                        # Git submodule (book source)
├── justfile                     # Task runner (run `just --list`)
├── docker-compose.yml           # Full stack (app, qdrant, postgres, build profiles)
├── docker-compose.dev.yml       # Dev override (source mount + hot-reload)
├── docker-compose.prod.yml      # GHCR image override (for production deployment)
├── .env.example                 # Config template (all available settings)
└── pyproject.toml
```

## License

Apache License 2.0 - see [LICENSE](LICENSE).
