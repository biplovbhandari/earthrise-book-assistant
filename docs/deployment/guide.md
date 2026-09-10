# Deployment Guide

Deploy the EarthRISE book assistant on a single machine.
This guide covers macOS (Mac Mini / Apple Silicon) and Linux deployment paths.

For model selection and memory budgets, see [deployment-models.md](../../system-design/deployment-models.md).

## Platform differences

The app, indexer, and reranker use ML models (sentence-transformers, fastembed) that benefit from GPU acceleration.
Docker containers on macOS cannot access the Metal GPU, so the app and indexer run natively on macOS.
On Linux with NVIDIA GPU, everything runs in Docker with GPU passthrough.

| Component | macOS | Linux (NVIDIA) |
|-----------|-------|----------------|
| Qdrant + Postgres | Docker | Docker |
| Ollama (LLM) | Native (Metal) | Native (CUDA) |
| App (uvicorn) | Native (Metal/MPS) | Docker (CUDA) |
| Indexer | Native (Metal/MPS) | Docker (CUDA) |
| Book builder | Docker (no GPU needed) | Docker |

## Prerequisites

### macOS note: admin vs non-admin users

Installing Docker Desktop and Ollama requires writing to `/Applications`, which needs an admin account.
If your deployment user is non-admin (recommended for servers), install from the admin account, then switch back to the non-admin user for everything else.

### just (task runner)

```bash
brew install just       # macOS
# Linux: see https://just.systems/man/en/installation.html
```

All deployment commands in this guide use `just` recipes.
Run `just --list` to see available commands.

### Docker

On macOS, install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
Download the `.dmg` from the website and drag Docker.app to `/Applications`, or:

```bash
brew install --cask docker-desktop
```

On Linux, install [Docker Engine](https://docs.docker.com/engine/install/) and the [Compose plugin](https://docs.docker.com/compose/install/linux/).
For GPU support, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

After installation, open the app once to complete setup:

```bash
open /Applications/Docker.app    # macOS
```

Docker Desktop runs in the background after first launch and auto-starts on boot.

### Ollama

Ollama runs natively (not in Docker) so it can access the GPU directly (Metal on macOS, CUDA on Linux).

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

On macOS, you can also download the `.dmg` from [ollama.com/download/mac](https://ollama.com/download/mac).

After installation, start Ollama before pulling models:

```bash
open /Applications/Ollama.app    # macOS, wait for menu bar icon
ollama serve &                   # Linux
```

Pull the LLM model:

```bash
ollama pull qwen3:8b
```

Verify it works:

```bash
ollama run qwen3:8b "Hello, what is semantic segmentation?"
```

You should get a coherent response.
Press Ctrl+D to exit.

### Python and uv (macOS only)

On macOS, the app and indexer run natively (not in Docker) to access the Metal GPU.
This requires Python and uv:

```bash
brew install python@3.12 uv
```

## 1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/biplovbhandari/earthrise-book-assistant.git
cd earthrise-book-assistant
```

## 2. Install dependencies (macOS only)

```bash
uv sync --group dev --group indexer
```

Not needed on Linux - the Docker images include all dependencies.

## 3. Configure the environment

```bash
cp .env.example .env
```

Edit `.env` with deployment settings.

**macOS (.env):**

```bash
LLM_MODEL=qwen3:8b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
DATABASE_URL=postgresql+asyncpg://earthrise:earthrise@localhost:5432/earthrise
RERANKER_PROVIDER=local_cross_encoder
RETRIEVAL_STRATEGY=hybrid
RATE_LIMIT_PER_MINUTE=30
MAX_CONCURRENT_LLM=1
```

On macOS, `LLM_BASE_URL` and `DATABASE_URL` use `localhost` because the app runs natively on the host alongside Ollama and Docker services.

**Linux (.env):**

```bash
LLM_MODEL=qwen3:8b
LLM_BASE_URL=http://172.17.0.1:11434/v1
LLM_API_KEY=ollama
DATABASE_URL=postgresql+asyncpg://earthrise:earthrise@postgres:5432/earthrise
RERANKER_PROVIDER=local_cross_encoder
RETRIEVAL_STRATEGY=hybrid
RATE_LIMIT_PER_MINUTE=30
MAX_CONCURRENT_LLM=1
```

On Linux, `LLM_BASE_URL` uses the Docker bridge gateway (`172.17.0.1`) and `DATABASE_URL` uses the Docker service name (`postgres`) because the app runs inside Docker.

## 4. Start services and application

### macOS

```bash
just services       # Docker: Qdrant + PostgreSQL
just db-migrate     # Apply database migrations
just dev            # Native: uvicorn with Metal GPU for embeddings
```

`just dev` starts the API with hot-reload.
The embedding model and reranker run on the Metal GPU via MPS.

### Linux

```bash
just up-prod        # Docker: full stack from GHCR images (app + Qdrant + PostgreSQL)
```

Database migrations run automatically on startup via the entrypoint script.
The app and its ML models use CUDA GPU inside the container.

## 5. Render the book and index content

### macOS

```bash
just render-book    # Docker: Quarto builder (no GPU needed)
just index          # Native: indexer with Metal GPU for embeddings
```

### Linux

```bash
just render-book-prod    # Docker: GHCR Quarto builder image
just index-prod          # Docker: GHCR indexer image with CUDA GPU
```

To rebuild the index from scratch, add `--recreate-collection`:

```bash
just index --recreate-collection          # macOS
just index-prod --recreate-collection     # Linux
```

## 6. Verify

```bash
# Health check
curl -s localhost:8000/health | python3 -m json.tool

# Search (retrieval only)
curl -s -X POST localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"question": "What is U-Net?"}' | python3 -m json.tool

# Chat (streaming, requires Ollama)
curl -N -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question": "What is semantic segmentation?"}'
```

Check that:
- `/health` shows `retrieval: ready`, `generation: ready`, `chat: ready`, `database: ready`
- `/search` returns chunks with scores
- `/chat` streams tokens via SSE

The book with chat widget is at http://localhost:8000/.
The Qdrant dashboard is at http://localhost:6333/dashboard.

## 7. Public access with Cloudflare Tunnel

Cloudflare Tunnel exposes the app to the internet with HTTPS, without opening ports on your router.

### Install cloudflared

```bash
# macOS
brew install cloudflare/cloudflare/cloudflared

# Linux
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  -o /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared
```

### Authenticate

```bash
cloudflared tunnel login
```

This opens a browser to authorize your Cloudflare account.

### Create a tunnel

```bash
cloudflared tunnel create earthrise
```

Note the tunnel ID from the output (a UUID like `abc123de-...`).

### Configure the tunnel

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: <TUNNEL_ID>
credentials-file: /Users/<you>/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: earthrise.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

Replace `<TUNNEL_ID>` with your tunnel ID and `earthrise.yourdomain.com` with your domain.

### Create DNS record

```bash
cloudflared tunnel route dns earthrise earthrise.yourdomain.com
```

### Start the tunnel

```bash
cloudflared tunnel run earthrise
```

The app is now reachable at `https://earthrise.yourdomain.com`.

### Run as a service (persistent)

To keep the tunnel running after logout:

```bash
# macOS
sudo cloudflared service install
sudo launchctl start com.cloudflare.cloudflared

# Linux (systemd)
sudo cloudflared service install
sudo systemctl enable --now cloudflared
```

## 8. Updating

When new code merges to main, GHCR images are rebuilt automatically.

### macOS

```bash
git pull
uv sync --group dev --group indexer
just render-book
just index
# Restart just dev (Ctrl+C and re-run)
```

### Linux

```bash
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull
just up-prod
just render-book-prod
just index-prod
```

Database migrations run automatically on restart via the entrypoint script.

## 9. Monitoring

Check application health:

```bash
curl -s localhost:8000/health | python3 -m json.tool
```

View logs:

```bash
# macOS: uvicorn logs are in the terminal running just dev
# Docker services:
docker compose logs -f qdrant     # vector DB logs
docker compose logs -f postgres   # database logs

# Linux (full Docker):
docker compose logs -f app        # API logs
docker compose logs -f qdrant     # vector DB logs
docker compose logs -f postgres   # database logs
```

Check Qdrant dashboard at http://localhost:6333/dashboard for collection stats and indexed chunk counts.

## 10. Maintenance

Reset the database without re-indexing (keeps Qdrant vectors intact):

```bash
just db-reset
```

Wipe everything for a completely fresh start (removes all volumes and data):

```bash
just clean
```
