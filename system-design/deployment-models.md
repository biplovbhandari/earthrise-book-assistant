# Deployment Model Selection

Model recommendations for self-hosting the EarthRISE book assistant.
All models run locally via Ollama (LLM) and HuggingFace/fastembed (embedding, reranker, sparse).

## Memory budget

The system runs five ML-heavy components alongside the OS and infrastructure services.

| Component | What it does | Memory |
|-----------|-------------|--------|
| **OS** | System services and file cache | ~3-4 GB |
| **Qdrant** | Vector database for retrieval | ~200-400 MB |
| **PostgreSQL** | Interaction recording and analytics | ~100-200 MB |
| **FastAPI app** | Python process, uvicorn, request handling | ~150-250 MB |
| **Embedding model** | Dense vector encoding for retrieval | 440 MB - 1.2 GB |
| **Sparse model** | Term-level retrieval (hybrid search) | 10 MB - 530 MB |
| **Reranker** | Re-scores retrieved chunks | 80 MB - 2.3 GB |
| **LLM** | Generates the answer from retrieved context | 3 - 20+ GB |

The embedding, sparse, and reranker models load into the FastAPI process.
The LLM runs in a separate Ollama process.
The LLM gets whatever memory is left after everything else.

In a RAG system, retrieval quality matters more than LLM size - a better embedding model puts better chunks in front of the LLM.

### Platform notes

On systems with **unified memory** (Apple Silicon), all components share one memory pool.
The memory tiers below assume this layout.

On systems with a **discrete GPU** (NVIDIA on Linux/Windows), the LLM can run in VRAM.
This frees system RAM for the other components and effectively increases the available budget.
Ollama uses CUDA automatically when an NVIDIA GPU is available.

Ollama runs natively on macOS, Linux, and Windows.
On macOS, run it outside Docker so it can access the Metal GPU.
On Linux with NVIDIA, Ollama can run in Docker with `--gpus all` or natively.

## Model inventory

### Embedding models

The embedding model determines the vector dimension stored in Qdrant and PostgreSQL.
Changing the model after indexing requires re-indexing all content and an Alembic migration for the `query_embedding` column.

| Model | Dimension | RAM | Quality (MTEB) | Notes |
|-------|-----------|-----|----------------|-------|
| `BAAI/bge-large-en-v1.5` | 1024 | ~1.2 GB | Higher | Current default. Best retrieval quality for English. |
| `BAAI/bge-base-en-v1.5` | 768 | ~440 MB | Good | Saves ~760 MB. Modest quality trade-off for domain-specific RAG. |
| `BAAI/bge-m3` | 1024 | ~2.2 GB | Higher | Multilingual. Only needed if book content is not English-only. |

### Sparse models (hybrid search)

Used alongside the embedding model for hybrid retrieval.
SPLADE expands query terms neurally; BM25 is statistical exact-match.

| Model | RAM | Quality | Notes |
|-------|-----|---------|-------|
| `prithivida/Splade_PP_en_v1` | ~530 MB | Better | Neural term expansion. Current default. |
| `Qdrant/bm25` | ~10 MB | Good | Statistical. Saves ~500 MB. Needs IDF modifier. |

### Reranker models

Re-scores the top-k retrieved chunks before sending to the LLM.

| Model | RAM | Quality | Notes |
|-------|-----|---------|-------|
| `cross-encoder/ms-marco-MiniLM-L6-v2` | ~80 MB | Good | Current default. Fast on CPU. |
| `BAAI/bge-reranker-v2-m3` | ~2.3 GB | Better | Multilingual, more accurate. Only justified on 24 GB+. |

### LLM models (via Ollama)

All models use Q4 quantization by default.

| Model | RAM (Q4) | Quality | Notes |
|-------|----------|---------|-------|
| `qwen3:8b` | ~5 GB | Good | Current default. Proven for RAG synthesis. |
| `qwen3.5:9b` | ~5.5 GB | Better | Newer generation, same memory class. |
| `qwen3.5:14b` | ~9 GB | Better | Needs 24 GB system. Noticeably better synthesis. |
| `mistral-small3.2:24b` | ~14 GB | Higher | Needs 24 GB system. Strong technical reasoning. |

## Recommended configurations

Memory tiers assume unified memory (Apple Silicon).
Systems with a discrete GPU can run larger LLMs since VRAM is separate from system RAM.

### 16 GB (tight but workable)

Budget: ~8-9 GB for LLM after OS (~4 GB) + services (~500 MB) + ML models (~1.8 GB).

```
EMBEDDING_MODEL_NAME=BAAI/bge-large-en-v1.5    # 1.2 GB, 1024-dim
EMBEDDING_DIMENSION=1024
SPARSE_MODEL_NAME=prithivida/Splade_PP_en_v1    # 530 MB
RERANKER_PROVIDER=local_cross_encoder
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L6-v2  # 80 MB
LLM_MODEL=qwen3:8b                              # 5 GB
MAX_CONCURRENT_LLM=1
```

**ML model total: ~6.8 GB.** Leaves ~5 GB for OS + Docker + KV cache.

If memory pressure occurs, two options ranked by impact:

1. Switch sparse model to `Qdrant/bm25` (saves ~500 MB, modest retrieval quality loss)
2. Switch embedding to `BAAI/bge-base-en-v1.5` (saves ~760 MB, requires re-indexing + migration to 768-dim)

### 24 GB (sweet spot)

Budget: ~16 GB for LLM.

```
EMBEDDING_MODEL_NAME=BAAI/bge-large-en-v1.5    # 1.2 GB, 1024-dim
EMBEDDING_DIMENSION=1024
SPARSE_MODEL_NAME=prithivida/Splade_PP_en_v1    # 530 MB
RERANKER_PROVIDER=local_cross_encoder
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L6-v2  # 80 MB
LLM_MODEL=qwen3.5:14b                           # 9 GB
MAX_CONCURRENT_LLM=2
```

**ML model total: ~10.8 GB.** Room for a larger LLM or `OLLAMA_NUM_PARALLEL=2`.

Alternative LLM: `mistral-small3.2:24b` (~14 GB) fits with less headroom.

### 32 GB+ (comfortable)

```
EMBEDDING_MODEL_NAME=BAAI/bge-large-en-v1.5    # 1.2 GB, 1024-dim
EMBEDDING_DIMENSION=1024
SPARSE_MODEL_NAME=prithivida/Splade_PP_en_v1    # 530 MB
RERANKER_PROVIDER=local_cross_encoder
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3     # 2.3 GB, better quality
LLM_MODEL=mistral-small3.2:24b                  # 14 GB
MAX_CONCURRENT_LLM=2
```

**ML model total: ~18 GB.** Room for multiple parallel LLM slots or a 31B model.

## Concurrency

Ollama processes LLM requests sequentially by default.
Set `OLLAMA_NUM_PARALLEL=2` to serve two concurrent users (doubles KV cache memory).

On 16 GB, keep `OLLAMA_NUM_PARALLEL=1`.
On 24 GB+, `OLLAMA_NUM_PARALLEL=2` works with the recommended configs above.

## Ollama setup

Install Ollama and pull the model:

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull qwen3:8b          # or whichever LLM you chose
```

When the FastAPI app runs inside Docker, set `LLM_BASE_URL` so the container can reach Ollama on the host:

- **macOS/Windows (Docker Desktop):** `http://host.docker.internal:11434/v1`
- **Linux:** `http://172.17.0.1:11434/v1` (Docker bridge gateway) or use `--network host`

## Embedding dimension and schema

The `Interaction.query_embedding` column dimension is driven by `EMBEDDING_DIMENSION` in `.env` (default 1024, matching `bge-large-en-v1.5`).
The startup check compares the embedder's actual output dimension against this setting and disables recording on mismatch.

Switching to a different dimension:

1. Update `EMBEDDING_MODEL_NAME` and `EMBEDDING_DIMENSION` in `.env`
2. Create an Alembic migration to alter the `query_embedding` column
3. Re-create the Qdrant collection with the new dimension
4. Re-index all book content
