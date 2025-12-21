# Deposium Embeddings - Instruction-Aware + Multi-Model Service

🔥 **NEW: Qwen25-1024D** - First instruction-aware static embeddings model! Quality: 0.841, Size: 65MB

Ultra-fast embeddings service with **instruction-awareness** + full-size models for maximum flexibility.

## 🚀 Features

- 🔥 **Qwen25-1024D** (PRIMARY) - Instruction-aware embeddings (UNIQUE capability)
- ⚡ **Gemma-768D** (SECONDARY) - Multilingual support
- 🎯 **EmbeddingGemma-300M** - Full-size embeddings (300M params)
- 🚀 **Qwen3-Embedding-0.6B** - Full-size embeddings (600M params, MTEB: 64.33)
- 🏆 **Qwen3 Reranking** - Optimized FP32 reranking (242ms)
- 📄 **Document Complexity Classifier** - Binary routing (LOW→OCR, HIGH→VLM)
- **500-1000x faster** than full LLMs
- **Ollama-compatible API** - drop-in replacement
- **FastAPI** backend with health checks

## 📊 Available Models

### 🔥 Qwen25-1024D (PRIMARY - INSTRUCTION-AWARE)
- **Quality:** 0.841 overall (+52% vs Gemma-768D)
- **Instruction-Awareness:** 0.953 (UNIQUE capability)
- **Semantic:** 0.950 | **Code:** 0.864 | **Conversational:** 0.846
- **Size:** 65MB (6x smaller than Gemma-768D)
- **Dimensions:** 1024D
- **Speed:** 500-1000x faster than full Qwen2.5-1.5B LLM
- **Base:** Qwen2.5-1.5B-Instruct (1.54B params distilled)
- **Use case:** Primary model - instruction-aware RAG, Q&A, code search, conversational AI

### ⚡ Gemma-768D (SECONDARY - MULTILINGUAL)
- **Quality:** 0.551 overall
- **Multilingual:** 0.737 (best for cross-language)
- **Size:** 400MB
- **Dimensions:** 768D
- **Speed:** 500-700x faster than full Gemma
- **Base:** google/embeddinggemma-300m
- **Use case:** Secondary - multilingual support, cross-language search

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

# Generate embedding with Qwen25-1024D (PRIMARY - instruction-aware)
curl -X POST http://localhost:11435/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen25-1024d","input":"Explain how neural networks work"}'

# Generate embedding with Gemma-768D (SECONDARY - multilingual)
curl -X POST http://localhost:11435/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-768d","input":"Machine learning et intelligence artificielle"}'

# Reranking with Qwen3 (FP32 optimized)
curl -X POST http://localhost:11435/api/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-rerank","query":"machine learning","documents":["AI and ML","cooking recipes","deep learning"]}'

# Document complexity classification (file upload)
curl -X POST http://localhost:11435/api/classify \
  -F "file=@document.jpg"

# Document complexity classification (base64 JSON)
curl -X POST http://localhost:11435/api/classify \
  -H "Content-Type: application/json" \
  -d '{"image":"data:image/jpeg;base64,/9j/4AAQSkZJRg..."}'
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
  "model": "qwen25-1024d",  // or "gemma-768d", "qwen3-embed"
  "input": "Explain how neural networks work"
}
```

**Response (qwen25-1024d - 1024D):**
```json
{
  "model": "qwen25-1024d",
  "embeddings": [[0.123, -0.456, ...]]  // 1024 dimensions
}
```

**Response (gemma-768d - 768D):**
```json
{
  "model": "gemma-768d",
  "embeddings": [[0.123, -0.456, ...]]  // 768 dimensions
}
```

### `POST /api/classify`
Classify document complexity for intelligent routing

**Method 1: File Upload**
```bash
curl -X POST http://localhost:11435/api/classify \
  -F "file=@document.jpg"
```

**Method 2: Base64 JSON**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Response:**
```json
{
  "class_name": "HIGH",
  "class_id": 1,
  "confidence": 0.982,
  "probabilities": {
    "LOW": 0.018,
    "HIGH": 0.982
  },
  "routing_decision": "Complex document - Route to VLM reasoning pipeline (~2000ms)",
  "latency_ms": 9.4
}
```

**Classification:**
- `LOW` - Simple documents (plain text, simple forms) → Route to OCR (~100ms)
- `HIGH` - Complex documents (charts, diagrams, tables) → Route to VLM reasoning (~2000ms)

**Model:** ResNet18 ONNX INT8 quantized (11MB)
**Performance:** 93% accuracy, 100% HIGH recall, ~10ms latency
**Lazy loading:** Model loads only on first request to save RAM

## 🔧 Integration with N8N

### Qwen25-1024D (PRIMARY - Instruction-Aware)
Configure N8N Ollama credentials:
- **Base URL:** `http://deposium-embeddings-turbov2:11435`
- **Model Name:** `qwen25-1024d`
- **Use case:** RAG with instruction queries, Q&A, code search

### Gemma-768D (SECONDARY - Multilingual)
Configure N8N Ollama credentials:
- **Base URL:** `http://deposium-embeddings-turbov2:11435`
- **Model Name:** `gemma-768d`
- **Use case:** Multilingual search, cross-language retrieval

### Document Complexity Classifier
Configure N8N HTTP Request node:
- **Method:** POST
- **URL:** `http://deposium-embeddings-turbov2:11435/api/classify`
- **Send Binary Data:** Enabled
- **Use case:** Intelligent routing - LOW complexity to OCR, HIGH complexity to VLM

**Workflow Example:**
```
1. [Trigger] Webhook receives document
2. [HTTP Request] POST to /api/classify
3. [Switch] Route based on class_name
   ├─ LOW → [Stirling PDF] Simple OCR (~100ms)
   └─ HIGH → [LLM Node] VLM reasoning (~2000ms)
4. [Return] Send response
```

## 📈 Performance

### Qwen25-1024D (PRIMARY)
- **Overall Quality:** 0.841 (+52% vs Gemma-768D)
- **Instruction-Awareness:** 0.953 (UNIQUE capability)
- **Inference Speed:** ~500-1000x faster than full Qwen2.5-1.5B LLM
- **Memory Usage:** ~100-150MB RAM
- **Model Size:** 65MB (10x smaller than Qwen3-Embedding)

### Gemma-768D (SECONDARY)
- **Overall Quality:** 0.551
- **Multilingual:** 0.737 (best for cross-language)
- **Inference Speed:** ~500-700x faster than full Gemma
- **Memory Usage:** ~200-300MB RAM
- **Model Size:** 400MB

### CPU Optimization
- Perfect for Railway 32 vCPU deployment
- Environment optimizations (OMP, jemalloc, KMP)
- FP32 models for best precision on vCPU

### 💾 Memory Optimizations (NEW)
- **Dynamic VRAM Management**: Lazy loading with LRU cache
- **Float16 Precision**: Qwen3-Reranker reduced from 4GB to ~1.2GB (70% reduction)
- **Auto-unloading**: Frees VRAM when limit exceeded (5GB default)
- **Priority System**: Keeps important models in memory
- **Total VRAM Usage**: ~2.1GB for all models (fits in 6GB GPU)

#### Memory Usage Breakdown:
- **Qwen25-1024D**: ~0MB GPU (Model2Vec runs on CPU)
- **Gemma-768D**: ~400MB GPU when loaded
- **Qwen3-Reranker**: ~1.2GB GPU (float16 optimized)
- **Classifier**: ~0MB GPU (ONNX runs on CPU)

## 🚀 Railway Deployment

```bash
railway up
```

Environment variables:
- `PORT=11435` (auto-detected)
- `HOST=0.0.0.0` (auto-configured)

## 🎯 OpenBench Benchmarking API - ✅ NEW (Sprint 62)

Standardized LLM benchmarking endpoints for quality evaluation.

### Endpoints

#### `GET /api/benchmarks`
List available benchmark categories and providers.

**Response:**
```json
{
  "categories": {
    "knowledge": { "name": "Knowledge", "description": "General knowledge", "benchmarks": ["MMLU", "TriviaQA"] },
    "coding": { "name": "Coding", "description": "Code generation", "benchmarks": ["HumanEval", "MBPP"] },
    "math": { "name": "Math", "description": "Mathematical reasoning", "benchmarks": ["GSM8K", "MATH"] },
    "reasoning": { "name": "Reasoning", "description": "Logic and deduction", "benchmarks": ["ARC", "HellaSwag"] },
    "cybersecurity": { "name": "Cybersecurity", "description": "Security tasks", "benchmarks": [] },
    "search": { "name": "Search", "description": "Retrieval quality", "benchmarks": [] }
  },
  "providers": ["groq", "openai", "anthropic"],
  "default_provider": "groq",
  "default_model": "llama-3.1-8b-instant"
}
```

#### `POST /api/benchmarks/run`
Run a standardized benchmark.

**Request:**
```json
{
  "category": "search",
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "sample_limit": 100
}
```

**Response:**
```json
{
  "category": "search",
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "score": 0.847,
  "metrics": {
    "precision": 0.82,
    "recall": 0.87,
    "f1": 0.845
  },
  "samples_evaluated": 100,
  "duration_seconds": 45.3,
  "timestamp": "2025-12-21T10:30:00Z",
  "errors": []
}
```

#### `POST /api/benchmarks/corpus-eval`
Evaluate a custom corpus for retrieval quality.

**Request:**
```json
{
  "corpus_data": [
    {
      "query": "What is machine learning?",
      "relevant_docs": ["Machine learning is a subset of AI..."],
      "context": "Technical documentation"
    }
  ],
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "sample_limit": 100
}
```

**Response:**
```json
{
  "category": "search",
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "score": 0.785,
  "metrics": {
    "retrieval_precision": 0.76,
    "semantic_similarity": 0.81
  },
  "samples_evaluated": 50,
  "duration_seconds": 23.1,
  "timestamp": "2025-12-21T10:35:00Z",
  "errors": []
}
```

### Implementation

**Files:**
- `src/benchmarks/__init__.py` - Module init
- `src/benchmarks/openbench_runner.py` - OpenBenchRunner class (469 lines)
- `src/main.py` - REST endpoints (lines 520-656)

**Dependencies:**
- `openbench>=0.1.0` - OpenBench framework (optional, falls back to simulated results)

**Integration with deposium_MCPs:**
```
deposium_MCPs (MCP tools) ──REST──► deposium_embeddings-turbov2 (OpenBench)
                                    └── /api/benchmarks/*
```

## 📚 References

- [Model2Vec](https://github.com/MinishLab/model2vec)
- [HuggingFace Model](https://huggingface.co/C10X/Qwen3-Embedding-TurboX.v2)
- [Ollama API Docs](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [OpenBench](https://openbench.dev) - LLM Benchmarking Framework
