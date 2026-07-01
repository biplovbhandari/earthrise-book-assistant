# earthrise-book-assistant

Book viewer and retrieval-augmented generation (RAG) assistant for the [EarthRISE Applied AI and Deep Learning Book](https://nasa-earthrise.github.io/EarthRISE-Applied-Artificial-Intelligence-and-Deep-Learning-Book/).
Serves Quarto-rendered chapters alongside a search and generation API.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (required for both paths -- runs Qdrant, or the full stack)
- [Ollama](https://ollama.com/) (for `/ask` generation -- pull a model with `ollama pull qwen3:8b`)
- [Python 3.12+](https://www.python.org/downloads/) (local setup only -- not needed for Docker deployment)
- [uv](https://docs.astral.sh/uv/) (local setup only)
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
Qdrant runs in Docker (or [build from source](https://qdrant.tech/documentation/installation/) with Rust).

```bash
# Install dependencies
uv sync --group dev --group indexer

# Configure .env for local dev
#   QDRANT_URL=http://localhost:6333
#   LLM_BASE_URL=http://localhost:11434/v1
#   LLM_API_KEY=ollama
#   LLM_MODEL=qwen3:8b

# Start Qdrant
docker compose up qdrant -d

# Transcribe YouTube lectures (optional -- skip if transcripts are already committed)
uv run --group indexer python scripts/transcribe.py

# Index the book (chapters + companion PDFs + transcripts if available)
QDRANT_URL=http://localhost:6333 HF_HOME=.cache/huggingface \
  BOOK_COMMIT_SHA=$(git -C book rev-parse HEAD) \
  uv run python scripts/index_book.py

# Start the app (requires Ollama running for /ask)
QDRANT_URL=http://localhost:6333 HF_HOME=.cache/huggingface \
  LLM_BASE_URL=http://localhost:11434/v1 LLM_API_KEY=ollama LLM_MODEL=qwen3:8b \
  uv run uvicorn api.main:app --reload
```

Try it:

```bash
# Health check (shows generation readiness)
curl -s localhost:8000/health | python3 -m json.tool

# Search (retrieval only)
curl -s -X POST localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"question": "What is U-Net?"}' | python3 -m json.tool

# Ask (generates an answer with citations -- requires Ollama)
curl -s -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "What is U-Net?"}' | python3 -m json.tool
```

If the index is empty or Qdrant is unreachable, `/search` and `/ask` return `503`.
If Ollama is not running, `/search` still works but `/ask` returns `503`.

## Index Book Content

Indexes book chapters, companion PDFs, and video transcripts.
Transcripts are committed to the repo, so cloning gives you everything -- no need to run `transcribe.py` first.

```bash
# Local
QDRANT_URL=http://localhost:6333 HF_HOME=.cache/huggingface \
  BOOK_COMMIT_SHA=$(git -C book rev-parse HEAD) \
  uv run python scripts/index_book.py

# Docker
BOOK_COMMIT_SHA=$(git -C book rev-parse HEAD) \
  docker compose --profile build run --rm indexer

# Fresh index (delete and recreate the Qdrant collection first)
# Add --recreate-collection to either command above, e.g.:
QDRANT_URL=http://localhost:6333 HF_HOME=.cache/huggingface \
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

```bash
# Setup
cp .env.example .env
# Edit .env -- set LLM_API_KEY and LLM_MODEL for /ask generation.
# QDRANT_URL and LLM_BASE_URL are overridden by docker-compose.yml
# so their .env values don't matter for Docker.

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
```

- Book: http://localhost:8000/
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

## Testing and Linting

```bash
uv sync --group dev
uv run pytest -v                                           # tests
uv run ruff check . && uv run ruff format --check .        # lint
uv run pyright                                             # type check
```

## Project Structure

```
earthrise-book-assistant/
├── src/earthrise_rag/           # Python package (RAG logic)
│   ├── config.py                # Pydantic BaseSettings, env-driven
│   ├── interfaces.py            # Shared protocols
│   ├── models/                  # Chunk, ScoredChunk, Document, Answer, Citation, IndexResult
│   ├── indexing/                # Parsers, chunkers, embedder, vector store, pipeline
│   ├── retrieval/               # DenseStrategy, NoOpReranker
│   ├── generation/              # LLM client, context builder, system prompt
│   ├── citations/               # Citation builder
│   └── query/                   # QueryPipeline (search + ask)
├── api/                         # FastAPI app
│   ├── main.py                  # /health, /ask, /search, static book
│   ├── dependencies.py          # Adapter wiring (factories, lazy imports)
│   └── routes/                  # /search and /ask endpoints
├── scripts/
│   ├── index_book.py            # CLI: index book + PDFs + transcripts into Qdrant
│   ├── transcribe.py            # CLI: download + transcribe YouTube lectures
│   └── run_query.py             # CLI: search the index
├── data/
│   ├── video_chapter_map.yml    # Maps video IDs to book chapters
│   └── transcripts/             # Whisper-generated JSON transcripts (committed)
├── notebooks/                   # Colab notebooks (GPU transcription)
├── widget/                      # Chat widget (injected into book pages)
├── infra/docker/                # Dockerfiles
├── tests/
├── system-design/               # Architecture docs
├── book/                        # Git submodule (book source)
├── docker-compose.yml
├── .env.example                 # Config template (all available settings)
└── pyproject.toml
```

## License

Apache License 2.0 -- see [LICENSE](LICENSE).
