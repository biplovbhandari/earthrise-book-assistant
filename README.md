# earthrise-book-assistant

Book viewer and RAG chat assistant for the [EarthRISE Applied AI and Deep Learning Book](https://nasa-earthrise.github.io/EarthRISE-Applied-Artificial-Intelligence-and-Deep-Learning-Book/). Serves Quarto-rendered chapters alongside a hybrid search chatbot.

## Getting Started

```bash
cd earthrise-book-assistant
git submodule update --init --recursive
cp .env.example .env
docker compose build app quarto-builder                  # build images
docker compose --profile build run --rm quarto-builder   # render the book
docker compose up -d                                     # start app + qdrant
```

- Book: http://localhost:8000/
- API health: http://localhost:8000/health
- Qdrant dashboard: http://localhost:6333/dashboard

To re-render the book after content changes:

```bash
docker compose --profile build run --rm quarto-builder
```

To stop everything:

```bash
docker compose down
```

## Development

Run qdrant in Docker, app locally with hot reload:

```bash
uv sync --group dev
docker compose up qdrant -d                    # vector DB only
uv run uvicorn api.main:app --reload           # app
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
├── configs/                     # Environment config templates
│   ├── local.env.example
│   └── prod.env.example
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
