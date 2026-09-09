# Deployment Guide

Deploy the EarthRISE book assistant on a local machine using Docker and Ollama.
This guide covers a single-machine setup suitable for a Mac Mini, Linux server, or similar host.

For model selection and memory budgets, see [deployment-models.md](../../system-design/deployment-models.md).

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [Ollama](https://ollama.com/) installed natively (not in Docker)

Ollama must run outside Docker so it can access the GPU directly (Metal on macOS, CUDA on Linux).

## 1. Install Ollama

### macOS

```bash
brew install ollama
```

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama service:

```bash
# macOS: Ollama runs as a background service after installation.
# Linux: start it manually or via systemd.
ollama serve &
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

## 2. Clone the repository

```bash
git clone --recurse-submodules https://github.com/biplovbhandari/earthrise-book-assistant.git
cd earthrise-book-assistant
```

## 3. Configure the environment

```bash
cp .env.example .env
```

Edit `.env` with production settings.
The key values to change:

```bash
# LLM - point to Ollama on the host
LLM_MODEL=qwen3:8b
LLM_BASE_URL=http://host.docker.internal:11434/v1    # macOS/Windows (Docker Desktop)
# LLM_BASE_URL=http://172.17.0.1:11434/v1            # Linux (Docker bridge gateway)
LLM_API_KEY=ollama

# Database - enable interaction recording
DATABASE_URL=postgresql+asyncpg://earthrise:earthrise@postgres:5432/earthrise

# Retrieval
RERANKER_PROVIDER=local_cross_encoder
RETRIEVAL_STRATEGY=hybrid

# Rate limiting and concurrency
RATE_LIMIT_PER_MINUTE=30
MAX_CONCURRENT_LLM=1    # set to 2 on 24GB+ systems
```

Docker Compose overrides `QDRANT_URL` and `DATABASE_URL` automatically so the containers can reach each other.
The `LLM_BASE_URL` must use `host.docker.internal` (macOS/Windows) or `172.17.0.1` (Linux) so the app container can reach Ollama on the host.

## 4. Build or pull images

### Option A: Pull pre-built images from GHCR (recommended)

Images are published on every merge to main.
Docker pulls the correct architecture (amd64 or arm64) automatically.

```bash
docker pull ghcr.io/biplovbhandari/earthrise-book-assistant/api:dev
docker pull ghcr.io/biplovbhandari/earthrise-book-assistant/indexer:dev
docker pull ghcr.io/biplovbhandari/earthrise-book-assistant/book-builder:dev
```

The repo includes `docker-compose.prod.yml` which overrides the build targets to use GHCR images.
Run with:

```bash
just up-prod
```

### Option B: Build locally

```bash
just build
```

Then run with:

```bash
just up
```

## 5. Render the book

```bash
just render-book
```

This renders the Quarto book with the chat widget injected and copies the output to the `book_html` Docker volume.

## 6. Index content

```bash
just index
```

This indexes book chapters, companion PDFs, and video transcripts into Qdrant.
You only need to re-index when the book content changes or the Qdrant volume is removed.

To start fresh:

```bash
just index --recreate-collection
```

## 7. Start the application

If you built locally:

```bash
just up
```

If using GHCR images:

```bash
just up-prod
```

Database migrations run automatically on startup via the entrypoint script.

## 8. Verify

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

## 9. Public access with Cloudflare Tunnel

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

## 10. Updating

When new code merges to main, GHCR images are rebuilt automatically.
To update the deployed app:

```bash
# Pull latest images and restart
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull
just up-prod

# Re-render book if widget or book content changed
just render-book

# Re-index if book content changed
just index
```

Database migrations run automatically on restart via the entrypoint script.

## 11. Monitoring

Check application health:

```bash
curl -s localhost:8000/health | python3 -m json.tool
```

View logs:

```bash
docker compose logs -f app        # API logs
docker compose logs -f qdrant     # vector DB logs
docker compose logs -f postgres   # database logs
```

Check Qdrant dashboard at http://localhost:6333/dashboard for collection stats and indexed chunk counts.
