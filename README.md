# Deposium Embeddings TurboV2

Ultra-fast embeddings, reranking, vision-language, and audio transcription service for production deployments.

## Features

- **15+ Models** - Embeddings, reranking, document classification, OCR, audio transcription
- **500-1000x faster** than full LLMs for embeddings
- **Audio Transcription** - Whisper via faster-whisper (4x faster than OpenAI)
- **Ollama-compatible API** - drop-in replacement
- **Dynamic VRAM Management** - lazy loading, LRU cache, auto-unloading
- **4-bit Quantization** - NF4 for rerankers (70% VRAM reduction)
- **FastAPI** backend with health checks

---

## Available Models

### Embeddings

| Model | Type | Dimensions | Quality | Speed | Use Case |
|-------|------|------------|---------|-------|----------|
| **m2v-bge-m3-1024d** | Model2Vec | 1024D | MTEB ~0.58 | 14k texts/s | PRIMARY - RAG, semantic search |
| **bge-m3-onnx** | ONNX INT8 | 1024D | MTEB ~0.60 | CPU optimized | High quality on CPU |
| **gemma-768d** | Model2Vec | 768D | 0.55 overall | 500x faster | LEGACY - multilingual |
| **qwen3-embed** | SentenceTransformer | 1024D | High | Float16 | Embeddings + reranking |
| **mxbai-embed-2d** | 2D Matryoshka | 1024D (24 layers) | Best | Full | Maximum quality |
| **mxbai-embed-2d-fast** | 2D Matryoshka | 768D (12 layers) | -15% | 2x faster | Balanced |
| **mxbai-embed-2d-turbo** | 2D Matryoshka | 512D (6 layers) | -20% | 4x faster | Speed priority |

### Reranking

| Model | Type | Languages | BEIR Score | VRAM | Use Case |
|-------|------|-----------|------------|------|----------|
| **mxbai-rerank-v2** | Cross-encoder (Qwen2) | 100+ | 55.57 | ~250MB (4-bit) | SOTA reranking |
| **mxbai-rerank-xsmall** | Cross-encoder (DeBERTa V1) | 100+ | ~50 | ~200MB | Fastest, lightweight |
| **qwen3-rerank** | Bi-encoder | Multi | High | ~1.2GB | Reranking alias |

### Vision-Language

| Model | Type | Size | Latency | Use Case |
|-------|------|------|---------|----------|
| **vl-classifier** | ResNet18 ONNX | 11MB | ~10ms | Document complexity routing |
| **lfm25-vl** | LFM2.5-VL-1.6B | 1.6B params | ~10-15s | Document OCR, text extraction |

### Audio Transcription

| Model | Type | RAM | WER | Speed | Use Case |
|-------|------|-----|-----|-------|----------|
| **whisper-tiny** | faster-whisper | ~40MB | 7.8% | Fastest | Quick transcription |
| **whisper-base** | faster-whisper | ~1GB | 5.0% | Balanced | **Default** - general use |
| **whisper-small** | faster-whisper | ~2GB | 3.4% | Good | Better accuracy |
| **whisper-medium** | faster-whisper | ~5GB | 2.9% | Slower | High accuracy |

**Features:**
- 4x faster than OpenAI Whisper (CTranslate2 backend)
- CPU/CUDA inference with INT8 quantization
- Automatic language detection (100+ languages)
- Word-level timestamps
- Voice Activity Detection (VAD)
- Translate to English

---

## API Endpoints

### `GET /health`
Health check endpoint

### `GET /api/tags`
List available models (Ollama-compatible)

### `POST /api/embed`
Generate embeddings

```bash
curl -X POST http://localhost:11435/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"m2v-bge-m3-1024d","input":"Your text here"}'
```

**Response:**
```json
{
  "model": "m2v-bge-m3-1024d",
  "embeddings": [[0.123, -0.456, ...]]
}
```

### `POST /api/rerank`
Rerank documents by relevance

```bash
curl -X POST http://localhost:11435/api/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mxbai-rerank-v2",
    "query": "machine learning",
    "documents": ["AI and ML", "cooking recipes", "deep learning"]
  }'
```

**Response:**
```json
{
  "model": "mxbai-rerank-v2",
  "results": [
    {"index": 0, "document": "AI and ML", "relevance_score": 5.18},
    {"index": 2, "document": "deep learning", "relevance_score": 2.94},
    {"index": 1, "document": "cooking recipes", "relevance_score": -3.15}
  ]
}
```

### `POST /api/classify`
Document complexity classification

```bash
# File upload
curl -X POST http://localhost:11435/api/classify \
  -F "file=@document.jpg"

# Base64 JSON
curl -X POST http://localhost:11435/api/classify \
  -H "Content-Type: application/json" \
  -d '{"image":"data:image/jpeg;base64,/9j/4AAQ..."}'
```

**Response:**
```json
{
  "class_name": "HIGH",
  "class_id": 1,
  "confidence": 0.982,
  "probabilities": {"LOW": 0.018, "HIGH": 0.982},
  "routing_decision": "Complex document - Route to VLM reasoning pipeline (~2000ms)",
  "latency_ms": 9.4
}
```

**Classification:**
- `LOW` - Simple documents (plain text) -> Route to OCR (~100ms)
- `HIGH` - Complex documents (charts, tables, diagrams) -> Route to VLM (~2000ms)

### `POST /api/vision`
Document OCR with LFM2.5-VL

```bash
curl -X POST http://localhost:11435/api/vision \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQ...",
    "model": "lfm25-vl",
    "prompt": "Extract all text from this document"
  }'
```

**Response:**
```json
{
  "model": "lfm25-vl",
  "response": "The document contains...",
  "latency_ms": 12500
}
```

### `POST /api/transcribe`
Audio transcription with Whisper

```bash
curl -X POST http://localhost:11435/api/transcribe \
  -H "X-API-Key: YOUR_KEY" \
  -F "file=@audio.mp3" \
  -F "model=whisper-base" \
  -F "language=en"
```

**Response:**
```json
{
  "model": "whisper-base",
  "text": "Hello, this is a test transcription.",
  "language": "en",
  "language_probability": 0.98,
  "duration": 5.0,
  "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "Hello..."}],
  "latency_ms": 1250
}
```

### `POST /api/audio/embed`
Audio embedding pipeline (transcribe + embed)

```bash
curl -X POST http://localhost:11435/api/audio/embed \
  -H "X-API-Key: YOUR_KEY" \
  -F "file=@podcast.mp3" \
  -F "whisper_model=whisper-base" \
  -F "embedding_model=m2v-bge-m3-1024d"
```

**Response:**
```json
{
  "model": "whisper-base",
  "text": "Transcribed text...",
  "language": "en",
  "embedding_model": "m2v-bge-m3-1024d",
  "embeddings": [[0.123, -0.456, ...]],
  "latency_ms": 2500
}
```

---

## Docker Usage

### Build & Run
```bash
docker build -t deposium-embeddings-turbov2 .
docker run -p 11435:11435 --gpus all deposium-embeddings-turbov2
```

### Docker Compose
```yaml
services:
  embeddings-turbov2:
    build: ./deposium_embeddings-turbov2
    ports:
      - "11435:11435"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Memory Optimizations

### Dynamic VRAM Management
- **Lazy loading**: Models load only on first request
- **LRU cache**: Least recently used models unloaded first
- **Auto-unloading**: Frees VRAM when limit exceeded (5GB default)
- **Priority system**: Important models stay in memory

### 4-bit Quantization (NF4)
- **mxbai-rerank-v2**: 1GB -> ~250MB (75% reduction)
- **qwen3-embed**: 4GB -> ~1.2GB (70% reduction)

Note: mxbai-rerank-xsmall (V1 DeBERTa) doesn't support 4-bit quantization but is already lightweight (~200MB).

### Memory Usage Breakdown
| Model | VRAM | Device |
|-------|------|--------|
| m2v-bge-m3-1024d | ~0MB | CPU (Model2Vec) |
| bge-m3-onnx | ~0MB | CPU (ONNX) |
| gemma-768d | ~400MB | GPU |
| qwen3-embed/rerank | ~1.2GB | GPU (Float16) |
| mxbai-embed-2d | ~800MB | GPU |
| mxbai-rerank-v2 | ~250MB | GPU (4-bit NF4) |
| mxbai-rerank-xsmall | ~200MB | CPU (DeBERTa V1) |
| lfm25-vl | ~3.2GB | GPU (BF16) |
| vl-classifier | ~0MB | CPU (ONNX) |

**Total**: ~2-3GB VRAM for typical usage (fits in 6GB GPU)

---

## N8N Integration

### Ollama Credentials
- **Base URL**: `http://deposium-embeddings-turbov2:11435`
- **Model Name**: `m2v-bge-m3-1024d` (or any available model)

### Timeout Configuration

**Important:** Models use lazy loading. The first request may take 30-60s while the model loads.

| Scenario | Recommended Timeout |
|----------|-------------------|
| First request (model loading) | 60-90s |
| Subsequent requests | 5-10s |
| VLM (lfm25-vl) | 60-120s |

For n8n-nodes-ollama-reranker:
- Go to **Additional Options** → **Request Timeout (ms)**
- Set to `60000` (60s) minimum for first requests
- Subsequent requests are fast (~100-300ms)

### Workflow Example (Document Routing)
```
1. [Trigger] Webhook receives document
2. [HTTP Request] POST to /api/classify
3. [Switch] Route based on class_name
   +-- LOW -> [Stirling PDF] Simple OCR (~100ms)
   +-- HIGH -> [HTTP Request] /api/vision with lfm25-vl (~10s)
4. [Return] Send response
```

---

## Benchmarks

### VL Classifier (ResNet18 distilled from CLIP)
| Metric | Value |
|--------|-------|
| Accuracy (test) | 100% (75/75) |
| HIGH Recall | 100% |
| LOW Recall | 100% |
| Model Size | 11.10 MB |
| Latency | 36-60ms |
| vs Old CLIP | 97% smaller, 2.5x faster |

### mxbai-rerank-v2 (SOTA Cross-Encoder)
| Benchmark | Score |
|-----------|-------|
| BEIR Average | 55.57 |
| Languages | 100+ |
| Quantization | 4-bit NF4 |

### mxbai-embed-2d (2D Matryoshka)
| Variant | Layers | Dimensions | Quality vs Full |
|---------|--------|------------|-----------------|
| Full | 24 | 1024D | 100% |
| Fast | 12 | 768D | ~85% |
| Turbo | 6 | 512D | ~80% |

---

## Environment Variables

```bash
# Model HuggingFace IDs (optional overrides)
HF_MODEL_M2V_BGE_M3=tss-deposium/m2v-bge-m3-1024d
HF_MODEL_BGE_M3_ONNX=gpahal/bge-m3-onnx-int8
HF_MODEL_GEMMA_768D=tss-deposium/gemma-deposium-768d
HF_MODEL_QWEN3_EMBED=Qwen/Qwen3-Embedding-0.6B
HF_MODEL_MXBAI_2D=mixedbread-ai/mxbai-embed-2d-large-v1
HF_MODEL_MXBAI_RERANK=mixedbread-ai/mxbai-rerank-base-v2
HF_MODEL_LFM25_VL=LiquidAI/LFM2.5-VL-1.6B

# Server config
PORT=11435
HOST=0.0.0.0
```

---

## References

- [CHANGELOG](CHANGELOG.md) - Release history
- [Model2Vec](https://github.com/MinishLab/model2vec)
- [mxbai-embed-2d](https://huggingface.co/mixedbread-ai/mxbai-embed-2d-large-v1)
- [mxbai-rerank-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2)
- [mxbai-rerank-xsmall](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1)
- [LFM2.5-VL](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B)
- [Ollama API Docs](https://github.com/ollama/ollama/blob/main/docs/api.md)
