# Deployment Status - EmbeddingGemma Integration

## ✅ Completed

### Code Changes
- [x] Integrated EmbeddingGemma-300m baseline into src/main.py:35-57
- [x] Added float16 quantization (1173MB → 587MB)
- [x] Fixed NaN bug with automatic float32 fallback (src/main.py:155-171)
- [x] Added sentence-transformers==3.3.1 dependency
- [x] Updated Dockerfile comments for EmbeddingGemma

### Documentation
- [x] Created `.env.example` with HF_TOKEN template
- [x] Created `HF_TOKEN_SETUP.md` with full setup instructions
- [x] Updated docker-compose.yml with HF_TOKEN environment variable

### Git Commits
- [x] commit `cbcf2dc` - Replace LEAF with EmbeddingGemma-300m baseline
- [x] commit `cc46529` - Detect and reject NaN embeddings (rejected approach)
- [x] commit `24a624d` - Auto fallback float32 when float16 produces NaN ✅
- [x] commit `e543c4d` - Add HF_TOKEN configuration instructions

### Performance Metrics
- **Model**: google/embeddinggemma-300m
- **Size**: 587MB (float16 GPU), 1.2GB (float32 fallback)
- **Dimensions**: 768D
- **Context**: 2048 tokens (preserved from original)
- **MTEB Score**: 0.80 Spearman
  - BIOSSES: 0.83
  - STSBenchmark: 0.88
- **Latency** (CPU, batch_size=8):
  - 512 tokens: ~15ms/embedding
  - 1024 tokens: ~40ms/embedding
  - 2048 tokens: ~115ms/embedding

## ⚠️ Pending - Manual Steps Required

### 1. Security - Revoke Exposed HF Token
```bash
# Token accidentally shared in logs:
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

# MUST DO:
1. Go to https://huggingface.co/settings/tokens
2. Revoke the exposed token immediately
3. Create NEW token with Read access
```

### 2. Local Docker Configuration

Edit `/home/nico/code_source/tss/deposium_fullstack/deposium-local/.env`:

```bash
# Line 150 (already added):
HF_TOKEN=your_new_hf_token_here
```

Then rebuild and restart:

```bash
cd /home/nico/code_source/tss/deposium_fullstack/deposium-local
docker-compose up -d --build embeddings-turbov2
```

### 3. Railway Configuration

**Option A: Railway Dashboard (Recommended)**
1. Go to Railway dashboard
2. Select project: `deposium_embeddings-turbov2`
3. Variables → Add Variable
4. Name: `HF_TOKEN`
5. Value: your_new_hf_token_here
6. Railway will auto-redeploy

**Option B: Railway CLI (if installed)**
```bash
railway login
railway link
railway variables set HF_TOKEN=your_new_hf_token_here
```

### 4. Accept HuggingFace Terms
Visit https://huggingface.co/google/embeddinggemma-300m and click "Agree and access repository"

## Testing

### Local API Test:
```bash
curl -X POST http://localhost:11435/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma", "input": "test embedding"}'
```

### Docker Test:
```bash
curl -X POST http://deposium-embeddings-turbov2:11435/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma", "input": "test embedding"}'
```

### Railway Test (after deployment):
```bash
curl -X POST https://deposium-embeddings-turbov2.railway.app/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma", "input": "test embedding"}'
```

## Expected Logs

### Success:
```
INFO:src.main:Loading EmbeddingGemma-300m baseline (768D, float16, 2048 tokens)...
INFO:sentence_transformers.SentenceTransformer:Use pytorch device_name: cuda:0
INFO:src.main:✅ EmbeddingGemma loaded!
INFO:src.main:   Device: cuda
INFO:src.main:   Max seq length: 2048 tokens
INFO:src.main:   Embedding dim: 768D
INFO:src.main:   Quantization: float16 (~587MB)
INFO:src.main:   MTEB Score: 0.80 Spearman (BIOSSES: 0.83, STSBenchmark: 0.88)
INFO:src.main:🚀 All models ready!
```

### Float32 Fallback (Normal):
```
WARNING:src.main:Float16 NaN detected, retrying with float32 for 5 texts
INFO:src.main:Float32 fallback successful for 5 texts
```

### Error - Missing Token:
```
requests.exceptions.HTTPError: 401 Client Error: Repository gated. You are trying to access a gated repo.
```

## Architecture

### Available Models:
1. **turbov2** - C10X/Qwen3-Embedding-TurboX.v2 (1024D, ~30MB, ultra-fast)
2. **int8** - C10X/int8 (256D, ~30MB, compact)
3. **gemma** - google/embeddinggemma-300m (768D, 587MB float16, high quality)

### Model Selection:
```json
{
  "model": "gemma",
  "input": "your text here"
}
```

### API Endpoints:
- `GET /` - Service info
- `GET /health` - Health check
- `GET /api/tags` - List available models
- `POST /api/embed` - Generate embeddings
- `POST /api/embeddings` - Alternative endpoint

## Next Steps (Future Optimization)

1. **INT8 Quantization** - Target ~200MB, evaluate quality loss
2. **ONNX Optimization** - CPU inference optimization
3. **Model2Vec Distillation** - Target ~30MB compact variant
4. **Multi-Variant Strategy** - Deploy multiple optimized versions

## Files Modified

- `src/main.py` - EmbeddingGemma integration + float32 fallback
- `requirements.txt` - Added sentence-transformers==3.3.1
- `Dockerfile` - Updated model download comments
- `.env.example` - HF_TOKEN template
- `HF_TOKEN_SETUP.md` - Full configuration guide
- `docker-compose.yml` (deposium-local) - Added HF_TOKEN environment variable

## Repository Links

- **GitHub**: https://github.com/theseedship/deposium_embeddings-turbov2
- **Branch**: main
- **Latest Commit**: e543c4d - Add HF_TOKEN configuration instructions
