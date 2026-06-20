# earthrise-book-assistant

Book viewer and RAG chat assistant for the [EarthRISE Applied AI and Deep Learning Book](https://nasa-earthrise.github.io/EarthRISE-Applied-Artificial-Intelligence-and-Deep-Learning-Book/). Serves Quarto-rendered chapters alongside a hybrid search chatbot.

## Getting Started

```bash
cd earthrise-book-assistant
git submodule update --init --recursive
cp .env.example .env
```

## Running with Docker

```bash
docker compose build app quarto-builder                  # build images
docker compose --profile build run --rm quarto-builder   # render the book
docker compose up -d                                     # start app + qdrant
```

- Book: http://localhost:8000/
- API health: http://localhost:8000/health
- Qdrant dashboard: http://localhost:6333/dashboard

To re-render after book content changes:

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
uv run uvicorn api.main:app --reload           # app with hot reload
uv run pytest -v                               # tests
```

## Project Structure

```
earthrise-book-assistant/
├── src/earthrise_rag/           # Python package (RAG logic)
│   ├── config.py                # Pydantic BaseSettings, env-driven
│   └── models/                  # Chunk, ScoredChunk, Document
├── api/                         # FastAPI app (thin handlers)
│   └── main.py                  # /health + static book serving
├── widget/                      # Chat widget (injected into book pages)
│   ├── chat.html                # Widget HTML
│   └── _quarto-chat.yml         # Quarto profile overlay
├── infra/docker/                # Dockerfiles + scripts
│   ├── Dockerfile.app
│   ├── Dockerfile.quarto
│   └── scripts/render_book.sh
├── tests/
├── system-design/               # Architecture docs
├── book/                        # Git submodule (book source)
├── docker-compose.yml           # App + Qdrant + quarto-builder
├── docker-compose.dev.yml       # Dev overrides (hot reload)
├── .env.example                 # Config template
└── pyproject.toml
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
