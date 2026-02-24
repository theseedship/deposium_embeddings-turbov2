# Security Audit & Hardening

Last audit: 2026-02-24
Status: Fully hardened (v3.2.1) - All security items resolved

---

## Fixed Issues

### CUDA Memory Leak (CRITICAL - Production)

**Symptom:** VRAM grew from 358 MB to 5807 MB over 29 hours with 0 models loaded in model_manager.

**Root cause:** CUDA tensors created during inference were never freed between requests. Three compounding factors:

1. No `torch.inference_mode()` around generate/encode calls - PyTorch retained computation graphs for backpropagation that would never happen
2. No `del inputs, outputs` - tensor references kept alive until Python GC (non-deterministic)
3. No `torch.cuda.empty_cache()` - CUDA allocator held freed blocks in its pool

**Fix applied to 4 files, 7 leak points:**

| File | Endpoint | Fix |
|------|----------|-----|
| `routes/vision.py` | `/api/vision` | `torch.inference_mode()` + `del` + `gc.collect()` + `empty_cache()` |
| `routes/vision.py` | `/api/vision/file` | Same |
| `routes/classification.py` | `/api/classify/file` (VLM path) | Same |
| `routes/classification.py` | `/api/classify/base64` (VLM path) | Same |
| `routes/reranking.py` | `/api/rerank` (SentenceTransformer path) | Same |
| `backends/huggingface.py` | `/v1/messages` (sync) | `del inputs, outputs, generated_ids` + `empty_cache()` |
| `backends/huggingface.py` | `/v1/messages` (stream) | `del inputs` + `empty_cache()` |

**Not affected:**
- `routes/embeddings.py` - `encode()` returns numpy arrays (CPU), no CUDA tensors
- `routes/audio.py` - Same (numpy)
- `routes/reranking.py` Model2Vec path - Same (numpy)
- `backends/vllm_local.py` - vLLM manages its own CUDA memory
- `backends/bitnet.py` - CPU only
- `classifier.py` - ONNX Runtime, CPU

---

### Remote Code Execution via trust_remote_code (CRITICAL - Security)

**Risk:** HuggingFace models can include arbitrary Python code that executes during loading. With `trust_remote_code=True` on all 13 model loading calls, a compromised model repository could steal API keys, exfiltrate data, or install backdoors.

**Fix:** Whitelist approach - only known model providers that require custom code are allowed:

```python
TRUSTED_REMOTE_CODE_PREFIXES = (
    "Qwen/",           # Qwen models use custom tokenizer/model code
    "LiquidAI/",       # LFM2.5-VL custom architecture
    "mixedbread-ai/",  # mxbai models use custom pooling
)
```

Override with `HF_TRUST_REMOTE_CODE=true` env var for new models during development.

---

### Unbounded Input (HIGH - DoS)

**Risk:** No size limits on request payloads allowed memory exhaustion attacks.

**Fix - Pydantic validation on all request schemas:**

| Schema | Field | Limit |
|--------|-------|-------|
| `EmbedRequest` | `input` | 256 texts max, 100K chars each |
| `RerankRequest` | `documents` | 1000 documents max |
| `RerankRequest` | `query` | 10K chars |
| `VisionRequest` | `image` (base64) | 50 MB |
| `VisionRequest` | `max_tokens` | 4096 |
| `AudioTranscribeRequest` | `audio` (base64) | 50 MB |

---

### Error Information Disclosure (MEDIUM)

**Risk:** `raise HTTPException(status_code=500, detail=str(e))` leaked internal file paths, model names, and stack traces to clients.

**Fix:** Generic error messages to clients, full details logged server-side:

```python
# Before (BAD)
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

# After (GOOD)
except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    logger.error(f"Error: {str(e)}", exc_info=True)
    raise HTTPException(status_code=500, detail="Processing failed")
```

---

## Fixed Items (v3.2.1)

### Rate Limiting (FIXED)

SlowAPI middleware with 200 req/min default per IP. Configured in `main.py`:

```python
from slowapi.middleware import SlowAPIMiddleware
app.state.limiter = shared.limiter  # Limiter(key_func=get_remote_address, default_limits=["200/minute"])
app.add_middleware(SlowAPIMiddleware)
```

Per-route overrides available via `@shared.limiter.limit("50/minute")` decorator.

### CORS Configuration (FIXED)

Env-configurable via `CORS_ALLOWED_ORIGINS` (comma-separated). Default: no cross-origin (empty list).

```bash
# Production
CORS_ALLOWED_ORIGINS=https://app.deposium.com,https://admin.deposium.com

# Development
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

### Blocking Sync in Async Handlers (FIXED)

All CPU/GPU-heavy operations offloaded to thread pool via `shared.run_sync()`:

| File | Blocking Call | Status |
|------|--------------|--------|
| `routes/embeddings.py` | `model.encode()` | Fixed |
| `routes/reranking.py` | `model.rank()`, `model.encode()`, `cos_sim()` | Fixed |
| `routes/vision.py` | `vlm.generate()`, `processor.apply_chat_template()` | Fixed (x2 endpoints) |
| `routes/classification.py` | `vlm.generate()` | Fixed (x2 VLM paths) |
| `routes/audio.py` | `handler.transcribe()`, `model.encode()` | Fixed (x3 endpoints) |

### Graceful Shutdown (FIXED)

`@app.on_event("shutdown")` handler in `main.py`:
1. Cancels background cleanup task
2. Unloads all loaded models
3. Clears CUDA cache

---

## Authentication

API key authentication via `EMBEDDINGS_API_KEY` env var. Supports:
- `X-API-Key` header
- `Authorization: Bearer <key>` header

When `EMBEDDINGS_API_KEY` is not set, auth is disabled with a warning log. For production, always set this variable.

---

## Model Security

- Models loaded from HuggingFace Hub use `safetensors` format by default (no pickle deserialization)
- `trust_remote_code` restricted to whitelisted prefixes
- Model names validated against registered `configs` dict (whitelist), preventing path traversal
- BitNet subprocess uses list-form (no shell injection), with timeout enforcement
