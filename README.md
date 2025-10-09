# Deposium Embeddings TurboX.v2

Ultra-fast static embeddings service using Model2Vec for CPU-optimized inference.

## 🚀 Features

- **500x faster** than traditional transformers on CPU
- **50x smaller** model size (~30MB vs 639MB)
- **Ollama-compatible API** - drop-in replacement
- **Static embeddings** - no GPU required
- **FastAPI** backend with health checks

## 📊 Model Info

- **Base Model:** C10X/Qwen3-Embedding-TurboX.v2
- **Type:** Static embeddings (Model2Vec)
- **Size:** ~30MB
- **Dimensions:** Auto-detected (likely 256-512)
- **Context:** 32K tokens max

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

# Generate embedding
curl -X POST http://localhost:11435/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"turbov2","input":"test embedding"}'
```

## 🔌 API Endpoints

### `GET /`
Service info and status

### `GET /health`
Health check endpoint

### `GET /api/tags`
List available models (Ollama-compatible)

### `POST /api/embed`
Generate embeddings

**Request:**
```json
{
  "model": "turbov2",
  "input": "your text here"
}
```

**Response:**
```json
{
  "model": "turbov2",
  "embeddings": [[0.123, -0.456, ...]]
}
```

## 🔧 Integration with N8N

Configure N8N Ollama credentials:
- **Base URL:** `http://deposium-embeddings-turbov2:11435`
- **Model Name:** `turbov2`

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
