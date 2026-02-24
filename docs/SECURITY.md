# Security Audit & Hardening

Last audit: 2026-02-24
Status: Hardened (v3.2.0)

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

## Remaining Items (Non-blocking)

### Rate Limiting (HIGH priority)

No rate limiting on any endpoint. Recommend adding `slowapi` or similar:

```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@router.post("/api/embed")
@limiter.limit("100/minute")
async def create_embedding(...):
```

### CORS Configuration (MEDIUM priority)

No CORS middleware configured. Add based on deployment context:

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-app.domain"],
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type"],
)
```

### Blocking Sync in Async Handlers (MEDIUM priority)

CPU-heavy operations (`encode()`, `generate()`) block the async event loop. Consider `run_in_executor` for high-concurrency scenarios:

```python
loop = asyncio.get_event_loop()
embeddings = await loop.run_in_executor(None, model.encode, texts)
```

### Graceful Shutdown (MEDIUM priority)

Background cleanup task not cancelled on SIGTERM. Add shutdown handler:

```python
@app.on_event("shutdown")
async def shutdown():
    app.state.cleanup_task.cancel()
    for name in list(model_manager.models.keys()):
        model_manager._unload_model(name)
```

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
