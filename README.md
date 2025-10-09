# Deposium Embeddings - Dual Model Service

Ultra-fast static embeddings service with **two models** using Model2Vec for CPU-optimized inference.

## 🚀 Features

- **Dual model support:** TurboX.v2 (1024D) + int8 (256D)
- **500x faster** than traditional transformers on CPU
- **50x smaller** model size (~30MB vs 639MB)
- **Ollama-compatible API** - drop-in replacement
- **Static embeddings** - no GPU required
- **FastAPI** backend with health checks

## 📊 Available Models

### TurboX.v2 (1024 dimensions)
- **HuggingFace:** C10X/Qwen3-Embedding-TurboX.v2
- **Base:** Qwen3-Embedding-0.6B
- **Type:** Static embeddings (Model2Vec)
- **Size:** ~30MB
- **Use case:** General-purpose embeddings, semantic search

### int8 (256 dimensions)
- **HuggingFace:** C10X/int8
- **Base:** Qwen3-Reranker-0.6B
- **Type:** Static embeddings (Model2Vec)
- **Size:** ~30MB
- **Use case:** Lightweight embeddings, reranking optimization

## 🐳 Docker Usage

### Build
```bash
docker build -t deposium-embeddings-turbov2 .
```

### Run
```bash
docker run -p 11435:11435 deposium-embeddings-turbov2
```

### Test
```bash
# Health check
curl http://localhost:11435/health

# List models
curl http://localhost:11435/api/tags

# Generate embedding with TurboX.v2 (1024D)
curl -X POST http://localhost:11435/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"turbov2","input":"test embedding"}'

# Generate embedding with int8 (256D)
curl -X POST http://localhost:11435/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"int8","input":"test embedding"}'
```

## 🔌 API Endpoints

### `GET /`
Service info and status

### `GET /health`
Health check endpoint

### `GET /api/tags`
List available models (Ollama-compatible)

### `POST /api/embed`
Generate embeddings with model selection

**Request:**
```json
{
  "model": "turbov2",  // or "int8"
  "input": "your text here"
}
```

**Response (turbov2 - 1024D):**
```json
{
  "model": "turbov2",
  "embeddings": [[0.123, -0.456, ...]]  // 1024 dimensions
}
```

**Response (int8 - 256D):**
```json
{
  "model": "int8",
  "embeddings": [[0.123, -0.456, ...]]  // 256 dimensions
}
```

## 🔧 Integration with N8N

### TurboX.v2 (1024D)
Configure N8N Ollama credentials:
- **Base URL:** `http://deposium-embeddings-turbov2:11435`
- **Model Name:** `turbov2`

### int8 (256D)
Configure N8N Ollama credentials:
- **Base URL:** `http://deposium-embeddings-turbov2:11435`
- **Model Name:** `int8`

## 📈 Performance

- **Inference Speed:** ~500x faster than qwen3-embedding:0.6b on CPU
- **Memory Usage:** ~100-200MB RAM
- **CPU Optimization:** Perfect for Railway 32 vCPU deployment

## 🚀 Railway Deployment

```bash
railway up
```

Environment variables:
- `PORT=11435` (auto-detected)
- `HOST=0.0.0.0` (auto-configured)

## 📚 References

- [Model2Vec](https://github.com/MinishLab/model2vec)
- [HuggingFace Model](https://huggingface.co/C10X/Qwen3-Embedding-TurboX.v2)
- [Ollama API Docs](https://github.com/ollama/ollama/blob/main/docs/api.md)
