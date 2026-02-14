# GPU Support for TurboV2

Conditional GPU support for TurboV2 embeddings and reranking via Docker build args and compose overrides. The Python code auto-detects CUDA — only Docker plumbing was needed.

## Prerequisites

- NVIDIA GPU (tested: RTX 4050 6 GiB VRAM, CUDA 13.1 driver)
- `nvidia-container-toolkit` installed
- Docker `nvidia` runtime configured

```bash
# Verify nvidia runtime is available
docker info | grep -i nvidia
# Expected: Runtimes: ... nvidia ...

# Verify GPU is visible
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

## How It Works

### Dockerfile (`ARG ENABLE_GPU`)

The Dockerfile installs PyTorch from a different index depending on the build arg:

| `ENABLE_GPU` | PyTorch Index | Wheel Size | Result |
|---|---|---|---|
| `false` (default) | `whl/cpu` | ~200 MiB | CPU-only, no CUDA libs |
| `true` | `whl/cu126` | ~2.5 GiB | CUDA 12.6 + NCCL + cuDNN |

The Python code (`model_manager.py`) already handles device detection:
- `torch.cuda.is_available()` → selects `cuda` or `cpu`
- 4-bit NF4 quantization for mxbai-rerank-v2 on CUDA
- VRAM tracking via `torch.cuda.memory_allocated()`

### Compose Override (`docker-compose.gpu.yml`)

Adds `runtime: nvidia`, GPU device reservation, and relaxed resource limits:

```yaml
services:
  embeddings-turbov2:
    build:
      args:
        ENABLE_GPU: "true"
    runtime: nvidia
    environment:
      NVIDIA_VISIBLE_DEVICES: all
      OMP_NUM_THREADS: 4    # Relaxed from 2 (CPU mode)
    deploy:
      resources:
        limits:
          memory: 4G         # Less RAM needed (GPU offloads inference)
          cpus: '4.0'        # More headroom for data prep
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Setup (Local with GPU)

### 1. Enable in `.env`

In `deposium-local/.env`:

```bash
ENABLE_GPU=true
COMPOSE_FILE=docker-compose.yml:docker-compose.override.yml:docker-compose.gpu.yml
TURBOV2_RERANKER_MODEL=mxbai-rerank-v2
```

### 2. Build and Start

```bash
cd deposium-local
docker compose build embeddings-turbov2
docker compose up -d embeddings-turbov2
```

### 3. Verify

```bash
# Check logs for "Device: cuda"
docker logs deposium-embeddings-turbov2 --tail 30

# Expected output:
# Device: cuda
# GPU Memory: 760MB used, 5161MB free (Total: 5921MB)
# ✅ mxbai-rerank-v2 loaded with 4-bit NF4 on CUDA (~75% VRAM reduction)
#    VRAM allocated: 436MB

# Check GPU usage
nvidia-smi

# Test embed
curl -s http://localhost:11436/api/embed \
  -H "Content-Type: application/json" \
  -d '{"input": ["test GPU"], "model": "m2v-bge-m3-1024d"}'

# Test rerank (4-bit on GPU)
curl -s http://localhost:11436/api/rerank \
  -H "Content-Type: application/json" \
  -d '{"model": "mxbai-rerank-v2", "query": "test", "documents": ["doc1", "doc2"]}'
```

## Environment Comparison

| Environment | GPU | Build Arg | Compose Files | Reranker | Image Size |
|---|---|---|---|---|---|
| **Local (GPU)** | RTX 4050 | `ENABLE_GPU=true` | base + override + gpu | mxbai-rerank-v2 (4-bit NF4) | ~4.5 GiB |
| **Railway staging** | None | `ENABLE_GPU=false` | N/A (Dockerfile only) | mxbai-rerank-xsmall | ~2 GiB |
| **Future GPU hosting** | Any CUDA 12.6+ | `ENABLE_GPU=true` | Adapted per platform | mxbai-rerank-v2 (4-bit NF4) | ~4.5 GiB |

## Performance Comparison

Measured on RTX 4050 (6 GiB VRAM) vs CPU-only (same host):

| Metric | CPU Mode | GPU Mode | Improvement |
|---|---|---|---|
| Rerank latency (3 docs) | ~3-5s | ~200-500ms | ~10x faster |
| CPU usage (inference) | 200%+ | <1% | CPU freed |
| RAM usage | 6 GiB limit | 4 GiB limit | -2 GiB |
| VRAM usage | 0 | ~1.4 GiB | mxbai-v2 + m2v |
| Reranker model | mxbai-rerank-xsmall | mxbai-rerank-v2 (SOTA) | Better quality |

## Disabling GPU

To revert to CPU-only:

```bash
# Option 1: Remove GPU compose file from COMPOSE_FILE
COMPOSE_FILE=docker-compose.yml:docker-compose.override.yml

# Option 2: Comment out both lines
# ENABLE_GPU=true
# COMPOSE_FILE=docker-compose.yml:docker-compose.override.yml:docker-compose.gpu.yml

# Then rebuild
docker compose build embeddings-turbov2
docker compose up -d embeddings-turbov2
```

## VRAM Budget (RTX 4050, 6 GiB)

| Model | VRAM (4-bit) | Priority | Loaded |
|---|---|---|---|
| m2v-bge-m3-1024d | ~100 MiB | 10 (highest) | Always |
| mxbai-rerank-v2 | ~436 MiB | 1 | On demand |
| bge-m3-onnx | ~150 MiB | 8 | On demand |
| qwen3-rerank | ~1.2 GiB | 8 | On demand |
| **Total active** | **~1.4 GiB** | | m2v + mxbai-v2 |
| **Available** | **~3.6 GiB** | | For other models |

VRAM limit is set to 5000 MiB (keeps 1 GiB margin). Models auto-unload after 180s of inactivity via `AUTO_UNLOAD_MODELS_TIME`.

## Files Modified

| File | Change |
|---|---|
| `Dockerfile` | `ARG ENABLE_GPU=false`, conditional `torch` install (cu126 vs cpu) |
| `requirements.txt` | Removed `torch>=2.8.0` (now in Dockerfile) |
| `deposium-local/docker-compose.gpu.yml` | NEW: GPU override (runtime, devices, build arg) |
| `deposium-local/.env` | `ENABLE_GPU=true`, `COMPOSE_FILE`, `TURBOV2_RERANKER_MODEL=mxbai-rerank-v2` |

## Troubleshooting

### "CUDA not available" in logs despite ENABLE_GPU=true

The image was built without GPU support. Rebuild:
```bash
docker compose build --no-cache embeddings-turbov2
```

### "nvidia runtime not found"

Install nvidia-container-toolkit:
```bash
# Ubuntu/Debian
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### OOM on GPU

Reduce VRAM limit or switch to a smaller reranker:
```bash
# In .env
VRAM_LIMIT_MB=3000
TURBOV2_RERANKER_MODEL=mxbai-rerank-xsmall  # Uses ~200 MiB instead of ~436 MiB
```
