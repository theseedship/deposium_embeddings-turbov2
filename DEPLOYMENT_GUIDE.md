# TurboX.v2 Deployment Guide

## 🚀 Quick Start - Local Testing

### 1. Build and Run Locally

```bash
cd /home/nico/code_source/tss/deposium_embeddings-turbov2

# Build Docker image
docker build -t deposium-embeddings-turbov2:latest .

# Run container
docker run -d \
  --name deposium-embeddings-turbov2-test \
  --network deposium-internal \
  -p 11435:11435 \
  deposium-embeddings-turbov2:latest

# Check health
curl http://localhost:11435/health
# Expected: {"status":"healthy"}

# Test embedding
curl -X POST http://localhost:11435/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"turbov2","input":"test embedding"}'
```

### 2. Test in N8N (Local)

**Configure Qwen Embedding Tool node:**
```
Base URL: http://deposium-embeddings-turbov2-test:11435
Model Name: turbov2
Instruction Type: [leave empty]
```

**Test workflow:**
1. Add "Qwen Embedding Tool" node
2. Set credentials (no auth needed)
3. Input text: "semantic search test"
4. Execute node
5. Should return 1024-dimension embedding in <50ms

## ☁️ Railway Deployment

### Option 1: Deploy from GitHub (Recommended)

#### Step 1: Push to GitHub
```bash
cd /home/nico/code_source/tss/deposium_embeddings-turbov2

# Initial commit
git add .
git commit -m "feat: TurboX.v2 embedding service - 500x faster CPU embeddings

- FastAPI service with Ollama-compatible API
- Model2Vec static embeddings (30MB)
- 1024D embeddings in <10ms
- Native batch support
- Railway-ready Dockerfile"

git push origin main
```

#### Step 2: Deploy to Railway
```bash
# Login to Railway
railway login

# Create new service
railway init

# Link to project
railway link

# Deploy from GitHub
railway up

# Or use Railway CLI to deploy directly
railway deploy
```

#### Step 3: Configure Railway Service
- **Port:** 11435
- **Health Check Path:** `/health`
- **Environment:** Production
- **Restart Policy:** Always
- **Resources:**
  - CPU: 0.5-1 vCPU (sufficient)
  - RAM: 256-512MB (model is 30MB)

### Option 2: Deploy from Docker Hub

#### Step 1: Build and Push Image
```bash
# Tag for Docker Hub
docker tag deposium-embeddings-turbov2:latest \
  theseedship/deposium-embeddings-turbov2:latest

# Push to Docker Hub
docker push theseedship/deposium-embeddings-turbov2:latest
```

#### Step 2: Deploy to Railway
- Create new service: "Deploy from Docker Image"
- Image: `theseedship/deposium-embeddings-turbov2:latest`
- Port: 11435
- Health check: `/health`

## 🔌 N8N Integration (Production)

### Step 1: Add Railway URL to N8N Credentials

**In N8N (deposium-n8n-primary):**
1. Go to Credentials
2. Add "Qwen Embedding Tool API" credentials:
   ```
   Base URL: https://your-service.railway.app
   Model Name: turbov2
   API Key: [leave empty - no auth]
   ```

### Step 2: Update Workflows

**Replace Ollama credentials with TurboX.v2:**
- Old: `http://deposium-ollama:11434` → `qwen3-embedding:0.6b`
- New: `https://your-service.railway.app` → `turbov2`

**Expected performance:**
- Latency: **<50ms** (vs 200-600ms with Qwen3)
- Batch: **Native support** (5-10x faster for multiple texts)
- Memory: **Minimal** (no VRAM needed)

### Step 3: Test Production Deployment

```bash
# Test Railway endpoint
RAILWAY_URL="https://your-service.railway.app"

# Health check
curl $RAILWAY_URL/health

# Generate embedding
curl -X POST $RAILWAY_URL/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"turbov2","input":"production test"}'

# Batch embedding
curl -X POST $RAILWAY_URL/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model":"turbov2",
    "input":["text 1","text 2","text 3"]
  }'
```

## 📊 Performance Monitoring

### Railway Metrics to Watch:
- **CPU usage:** Should stay <30% for normal load
- **Memory:** Should stay <200MB
- **Response time:** Should be <50ms p95
- **Errors:** Should be 0%

### N8N Performance Improvement:
- **Before (Qwen3):** 200-600ms per embedding
- **After (TurboX.v2):** 5-50ms per embedding
- **Speedup:** **10-100x faster**

## 🔐 Security (Optional)

### Add API Key Authentication

**Update `src/main.py`:**
```python
from fastapi import Header, HTTPException

API_KEY = os.getenv("API_KEY", "")

async def verify_api_key(x_api_key: str = Header(...)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.post("/api/embed", dependencies=[Depends(verify_api_key)])
async def create_embedding(request: EmbedRequest):
    # ... existing code
```

**Railway Environment Variable:**
```
API_KEY=your-secure-random-key
```

## 🛠️ Troubleshooting

### Service won't start
```bash
# Check Railway logs
railway logs

# Common issues:
# - Port 11435 not exposed → Check Dockerfile EXPOSE
# - Model download failed → Check internet connectivity
# - Python deps missing → Rebuild with --no-cache
```

### High latency on Railway
```bash
# Check Railway region
railway status

# Use same region as N8N deployment
# Latency increases with distance
```

### Model not loading
```bash
# Check Railway logs for:
# "Loading TurboX.v2 model..."
# "✅ Model loaded successfully!"

# If stuck, check:
# - HuggingFace access (model downloads from HF)
# - Disk space (30MB needed)
```

## 🎯 Next Steps

### After Successful Deployment:

1. **Update N8N workflows** to use TurboX.v2
2. **Monitor performance** for 24-48 hours
3. **Compare embedding quality** for your use case
4. **Scale if needed** (Railway auto-scales)
5. **Document savings** (cost, latency, throughput)

### Cost Optimization:

**Railway cost comparison:**
- Qwen3 + Ollama: ~$10-20/month (1-2GB RAM, GPU optional)
- TurboX.v2: **~$2-5/month** (256-512MB RAM, CPU only)
- **Savings: 50-75%** with better performance

## 📚 API Reference

### Endpoints

#### `GET /`
Service information
```json
{
  "service": "Deposium Embeddings TurboX.v2",
  "status": "running",
  "model": "C10X/Qwen3-Embedding-TurboX.v2"
}
```

#### `GET /health`
Health check
```json
{"status": "healthy"}
```

#### `GET /api/tags`
List available models (Ollama-compatible)
```json
{
  "models": [
    {
      "name": "turbov2",
      "size": 30000000,
      "digest": "turbov2",
      "modified_at": "2025-10-09T00:00:00Z"
    }
  ]
}
```

#### `POST /api/embed`
Generate embeddings (Ollama-compatible)
```json
{
  "model": "turbov2",
  "input": "text" | ["text1", "text2"]
}
```

Response:
```json
{
  "model": "turbov2",
  "embeddings": [[0.1, 0.2, ...]]  // 1024 dimensions
}
```

## 🔗 Related Links

- [TurboX.v2 Model](https://huggingface.co/C10X/Qwen3-Embedding-TurboX.v2)
- [Model2Vec Documentation](https://github.com/MinishLab/model2vec)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Railway Deployment](https://docs.railway.app/)
