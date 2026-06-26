# earthrise-book-assistant

Book viewer and RAG chat assistant for the [EarthRISE Applied AI and Deep Learning Book](https://nasa-earthrise.github.io/EarthRISE-Applied-Artificial-Intelligence-and-Deep-Learning-Book/). Serves Quarto-rendered chapters alongside a dense retrieval search API.

## Getting Started

```bash
git submodule update --init --recursive
cp .env.example .env
# Edit .env — set QDRANT_URL=http://localhost:6333 for local dev
```

## Running with Docker

```bash
docker compose build app quarto-builder indexer            # build images
docker compose --profile build run --rm quarto-builder     # render the book
docker compose up -d                                       # start app + qdrant
```

- Book: http://localhost:8000/
- API health: http://localhost:8000/health
- Search API: http://localhost:8000/search
- Qdrant dashboard: http://localhost:6333/dashboard

### Index book content

```bash
# Docker
BOOK_COMMIT_SHA=$(git -C book rev-parse HEAD) docker compose --profile build run --rm indexer

# Local
uv sync --group indexer
QDRANT_URL=http://localhost:6333 HF_HOME=.cache/huggingface BOOK_COMMIT_SHA=$(git -C book rev-parse HEAD) uv run python scripts/index_book.py
```

Check the Qdrant dashboard for indexed chunks. Logs written to `logs/`.

### Search the index

```bash
# CLI
QDRANT_URL=http://localhost:6333 HF_HOME=.cache/huggingface uv run python scripts/run_query.py "What is U-Net?"
QDRANT_URL=http://localhost:6333 HF_HOME=.cache/huggingface uv run python scripts/run_query.py "crop mapping" --top-k 5 --filter chapter=03_Semantic_Segmentation

# API
curl -s -X POST localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"question": "What is U-Net?"}' | python3 -m json.tool

# With filters and top_k
curl -s -X POST localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"question": "crop mapping", "top_k": 5, "filters": {"chapter": "03_Semantic_Segmentation"}}' | python3 -m json.tool
```

If the index is empty or Qdrant is unreachable, `/search` returns `503`.

### Re-render book

```bash
docker compose --profile build run --rm quarto-builder
```

Code changes require `docker compose build app`. For hot reload in Docker:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

To stop: `docker compose down`

## Local Development

```bash
uv sync --group dev
docker compose up qdrant -d                    # vector DB only
QDRANT_URL=http://localhost:6333 HF_HOME=.cache/huggingface uv run uvicorn api.main:app --reload  # app with hot reload
uv run pytest -v                               # tests
uv run ruff check . && uv run ruff format --check .   # lint
```

## Project Structure

```
earthrise-book-assistant/
├── src/earthrise_rag/           # Python package (RAG logic)
│   ├── config.py                # Pydantic BaseSettings, env-driven
│   ├── interfaces.py            # Shared protocols (Embedder, VectorStore, RetrievalStrategy, Reranker)
│   ├── models/                  # Chunk, ScoredChunk, Document, IndexResult
│   ├── indexing/                # Parsers, chunkers, embedder, vector store, pipeline
│   ├── retrieval/               # DenseStrategy, NoOpReranker
│   └── query/                   # QueryPipeline (search, and later ask)
├── api/                         # FastAPI app (thin handlers)
│   ├── main.py                  # /health, /search, static book serving
│   ├── dependencies.py          # Adapter wiring (two factories, lazy imports)
│   └── routes/
│       └── search.py            # POST /search endpoint
├── scripts/
│   ├── index_book.py            # CLI: index book content into Qdrant
│   └── run_query.py             # CLI: search the index
├── widget/                      # Chat widget (injected into book pages)
│   ├── chat.html
│   └── _quarto-chat.yml         # Quarto profile overlay
├── infra/docker/                # Dockerfiles + scripts
│   ├── Dockerfile.app
│   ├── Dockerfile.quarto
│   ├── Dockerfile.indexer
│   └── scripts/render_book.sh
├── tests/
├── system-design/               # Architecture docs
├── book/                        # Git submodule (book source)
├── docker-compose.yml           # App + Qdrant + quarto-builder + indexer
├── docker-compose.dev.yml       # Dev overrides (hot reload)
├── .env.example                 # Config template
└── pyproject.toml
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
