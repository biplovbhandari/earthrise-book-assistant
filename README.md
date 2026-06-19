# earthrise-book-assistant

Book viewer and RAG chat assistant for the [EarthRISE Applied AI and Deep Learning Book](https://nasa-earthrise.github.io/EarthRISE-Applied-Artificial-Intelligence-and-Deep-Learning-Book/). Serves Quarto-rendered chapters alongside a hybrid search chatbot.

## Development

Run qdrant in Docker, app locally with hot reload:

```bash
cp .env.example .env
uv sync --group dev
docker compose up qdrant -d                    # vector DB only
uv run uvicorn api.main:app --reload           # app
uv run pytest -v                               # tests
```

- App: http://localhost:8000/health
- Qdrant dashboard: http://localhost:6333/dashboard

## Full Docker

Run everything in containers (no local Python needed):

```bash
cp .env.example .env
docker compose up -d                           # app + qdrant
curl http://localhost:8000/health
docker compose down                            # stop
```

Code changes require `docker compose build` to take effect. For hot reload in Docker, use the dev override:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## Project Structure

```
earthrise-book-assistant/
├── src/earthrise_rag/           # Python package (RAG logic)
│   ├── config.py                # Pydantic BaseSettings, env-driven
│   └── models/                  # Chunk, ScoredChunk, Document
├── api/                         # FastAPI app (thin handlers)
│   └── main.py                  # /health endpoint
├── infra/docker/                # Dockerfiles
│   └── Dockerfile.app
├── configs/                     # Environment config templates
│   ├── local.env.example
│   └── prod.env.example
├── tests/
├── system-design/               # Architecture docs
├── book/                        # Git submodule (book content)
├── docker-compose.yml           # App + Qdrant
├── docker-compose.dev.yml       # Dev overrides (hot reload)
├── .env.example                 # Config template
└── pyproject.toml
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
