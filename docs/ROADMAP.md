# 🗺️ ROADMAP - FastAPI Optimization & Scaling Strategy

**Project:** Deposium Embeddings FastAPI Server
**Version:** 10.0.0
**Date:** 2025-10-20
**Status:** Phase 0 (Current baseline)

---

## 📋 Executive Summary

### Current Situation

**Performance:**
- Single embedding: 1-5ms (Model2Vec)
- Batch 10 texts: 8-15ms
- Rerank 3 docs: 242ms (Qwen3-FP32)
- Theoretical throughput: 200-1000 req/sec (no queue)

**Infrastructure:**
- Railway deployment
- RAM usage: ~3GB (5 models loaded)
- CPU usage: <20% (I/O-bound, not CPU-bound)
- Available RAM: Up to 32GB (want to stay ≤4GB for cost)
- Current traffic: 0 concurrent users

**Models Loaded:**
```
qwen25-1024d (Model2Vec):        ~65MB  ← PRIMARY (instruction-aware)
gemma-768d (Model2Vec):         ~400MB  ← SECONDARY (multilingual)
qwen3-embed (SentenceTransf):   ~600MB  ← Full-size embeddings
embeddinggemma-300m:            ~300MB  ← Full-size Gemma
qwen3-rerank (shared):             0MB  ← Reranking
────────────────────────────────────────
Total models:                   ~1.4GB
Runtime overhead:              ~1.6GB
════════════════════════════════════════
TOTAL RAM:                       ~3GB
```

### Strategy: Latency-First → Progressive Concurrency Scaling

**Rationale:**
1. **Current bottleneck:** No caching (recalculates everything)
2. **Cost optimization:** Stay under 4GB until real traffic demands scale
3. **User experience:** Ultra-fast single requests > mediocre concurrent requests
4. **Progressive investment:** Pay for infrastructure as you grow

**Goals:**
- Phase 1: Optimize single-request latency (-90% with caching) at +$0 cost
- Phase 2: Handle 10-50 concurrent users at +$5/month
- Phase 3: Scale to 100+ concurrent users at +$20/month

---

## 🔍 Bottlenecks Analysis

### Current Architecture Issues

#### HIGH PRIORITY - Major Impact (50-300% gains possible)

**1. No Response Caching ⭐⭐⭐⭐⭐**
```python
# Current: main.py:275
embeddings = selected_model.encode(texts, show_progress_bar=False)
# ❌ ALWAYS recalculates, even for identical texts
```

**Problem:** Repeated queries (common in RAG/FAQ) recalculate embeddings every time.

**Impact:**
- 70-90% latency reduction for cache hits
- Typical cache hit rate in production: 30-60%
- Example: "What is the status?" asked 100x → 99x instant response

**Solution:** LRU cache with text hash key
**Cost:** +12MB RAM (1000 embeddings × 4KB each × 3 models)

---

**2. Single Worker Process ⭐⭐⭐⭐**
```dockerfile
# Current: Dockerfile:60
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "11435"]
# ❌ ONE process = ONE request at a time
```

**Problem:** Cannot handle concurrent requests efficiently.

**Impact:**
- 2-4x throughput with multi-worker
- BUT: +100% RAM per worker (models duplicated!)

**Solution:** Multi-worker with lazy loading (Phase 3)
**Cost:** +2-3GB RAM (see Phase 3 for details)

**Why NOT now:**
- 0 concurrent users currently
- Would cost +$15/month for unused capacity
- Better to optimize single-request latency first

---

**3. No Batch Processing Endpoint ⭐⭐⭐⭐**
```python
# Current: main.py:257
@app.post("/api/embed")
# ❌ Processes ONE request at a time, no bulk optimization
```

**Problem:** Indexing 1000 documents = 1000 separate API calls.

**Impact:**
- 3-5x efficiency for bulk operations
- Example: Index 1000 docs: 1000 requests → 1 batch request
- Reduces network overhead, connection setup time

**Solution:** Add `/api/embed/batch` endpoint
**Cost:** +0MB RAM (just code optimization)

---

**4. Manual Cosine Similarity ⭐⭐⭐**
```python
# Current: main.py:334-337
scores = [
    np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
    for doc_emb in doc_embs
]
# ❌ Python loop, not vectorized
```

**Problem:** Slow reranking (Python loop instead of optimized C/BLAS).

**Impact:**
- 2-3x faster reranking
- 242ms → ~100ms for 3 docs

**Solution:** Use sklearn's optimized `cosine_similarity()`
**Cost:** +0MB RAM (already have sklearn via sentence-transformers)

---

#### MEDIUM PRIORITY - Moderate Impact (10-50% gains)

**5. No Request Coalescing ⭐⭐⭐**

**Problem:** Multiple identical concurrent requests processed separately.

**Impact:** 30-50% reduction for concurrent identical queries

**Solution:** Request deduplication with asyncio locks
**Cost:** +50MB RAM (pending request cache)

---

**6. No Preloaded Common Embeddings ⭐⭐⭐**

**Problem:** FAQ-style queries recalculated on first request.

**Impact:** 40-60% latency for first request on common queries

**Solution:** Preload top 100 common queries at startup
**Cost:** +40MB RAM (100 queries × 4KB × 3 models)

---

#### LOW PRIORITY - Minor Impact (<10% gains)

**7. Logging Overhead ⭐**

**Problem:** `logger.info()` on every request (main.py:280, 356).

**Impact:** 1-3% latency reduction

**Solution:** Use `logger.debug()` for per-request logs
**Cost:** +0MB RAM

---

### Python 3.14 GIL-Free Evaluation

**Question:** Should we upgrade to Python 3.14 for GIL-free multithreading?

**Analysis:**
- Current bottlenecks: I/O-bound (network) and cache misses
- Model inference: Already GIL-free (numpy/PyTorch release GIL for C extensions)
- AsyncIO: Already handles I/O concurrency efficiently

**Estimated Impact:** <5% performance gain

**Recommendation:** ❌ **NOT RECOMMENDED**
- Low ROI (<5% gain)
- Unstable (Python 3.14 just released, ecosystem not ready)
- Better optimizations available (caching = 70-90% gain!)

**Revisit when:**
- Python 3.14 is stable (6-12 months)
- All dependencies support it (transformers, torch, etc.)
- After implementing Phase 1-3 optimizations

---

## 🚀 Phase 1: Quick Wins (0-10 req/sec)

**Timeline:** 1 week
**Target Users:** 0-10 concurrent users
**Cost Impact:** +$0/month (stay in same Railway tier)

### Objectives

1. ✅ Optimize single-request latency (5ms → <1ms for cache hits)
2. ✅ Add bulk processing capability (1000 requests → 1 batch)
3. ✅ Protect against OOM during traffic spikes
4. ✅ Add monitoring to know when to scale

### Optimizations

#### 1.1 LRU Cache for Embeddings

**Implementation:**
```python
from functools import lru_cache
import hashlib

# Cache configuration (configurable via env)
CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))

@lru_cache(maxsize=CACHE_SIZE)
def _get_cached_embedding(model_name: str, text_hash: str, text: str):
    """
    Cache embeddings by text hash.

    Args:
        model_name: Model identifier
        text_hash: MD5 hash of text (for cache key)
        text: Original text (for actual encoding)

    Returns:
        numpy.ndarray: Embedding vector
    """
    return models[model_name].encode([text], show_progress_bar=False)[0]

def encode_with_cache(model_name: str, texts: List[str],
                      max_cache_length: int = 500) -> List[np.ndarray]:
    """
    Encode texts with optional caching for short texts.

    Args:
        model_name: Model to use
        texts: List of texts to encode
        max_cache_length: Cache only texts shorter than this (default 500 chars)

    Returns:
        List of embedding arrays
    """
    embeddings = []

    for text in texts:
        if len(text) <= max_cache_length:
            # Short text: use cache
            text_hash = hashlib.md5(text.encode()).hexdigest()
            emb = _get_cached_embedding(model_name, text_hash, text)
        else:
            # Long text: compute directly (no cache)
            emb = models[model_name].encode([text], show_progress_bar=False)[0]

        embeddings.append(emb)

    return embeddings

# Usage in endpoint
@app.post("/api/embed")
async def create_embedding(request: EmbedRequest):
    texts = [request.input] if isinstance(request.input, str) else request.input
    embeddings = encode_with_cache(request.model, texts)
    # ... return response
```

**Configuration:**
```bash
# .env or Railway environment variables
EMBEDDING_CACHE_SIZE=1000        # Number of embeddings to cache
EMBEDDING_MAX_CACHE_LENGTH=500   # Only cache texts <500 chars
```

**RAM Impact:**
```
1 embedding (qwen25-1024d):  1024 dims × 4 bytes = 4KB
Cache 1000 embeddings:       1000 × 4KB = 4MB per model
3 models with cache:         3 × 4MB = 12MB total
════════════════════════════════════════════════════
Total Phase 1.1:            +12MB RAM
```

**Expected Gains:**
- Cache hit rate (typical): 30-60%
- Latency with cache hit: 5ms → <0.5ms (-90%)
- Weighted average: 5ms → 2-3ms (-40-60%)

---

#### 1.2 Batch Embedding Endpoint

**Implementation:**
```python
class BatchEmbedRequest(BaseModel):
    model: str = "qwen25-1024d"
    inputs: List[List[str]]  # [[batch1], [batch2], ...]
    batch_size: int = 32     # Process 32 texts at once
    use_cache: bool = True   # Enable caching

class BatchEmbedResponse(BaseModel):
    model: str
    batches: List[List[List[float]]]  # [[[emb1], [emb2]], [[emb3], [emb4]]]
    cache_stats: dict  # {"hits": 150, "misses": 50, "hit_rate": 0.75}

@app.post("/api/embed/batch")
async def create_batch_embeddings(request: BatchEmbedRequest):
    """
    Process multiple embedding batches efficiently.

    Use case: Indexing large document collections.
    Example: Index 1000 documents in 1 request instead of 1000 requests.

    Returns:
        Batches of embeddings preserving original structure
    """
    if request.model not in models:
        raise HTTPException(status_code=400, detail=f"Model not found")

    try:
        cache_hits = 0
        cache_misses = 0
        all_embeddings = []

        # Flatten all inputs for efficient processing
        all_texts = [text for batch in request.inputs for text in batch]

        # Process with caching if enabled
        if request.use_cache:
            for text in all_texts:
                text_hash = hashlib.md5(text.encode()).hexdigest()
                try:
                    emb = _get_cached_embedding(request.model, text_hash, text)
                    cache_hits += 1
                except:
                    emb = models[request.model].encode([text], show_progress_bar=False)[0]
                    cache_misses += 1
                all_embeddings.append(emb)
        else:
            # No caching: batch process all
            for i in range(0, len(all_texts), request.batch_size):
                batch = all_texts[i:i + request.batch_size]
                embs = models[request.model].encode(batch, show_progress_bar=False)
                all_embeddings.extend(embs)

        # Reconstruct original batch structure
        results = []
        idx = 0
        for batch in request.inputs:
            batch_embs = [all_embeddings[idx + i].tolist() for i in range(len(batch))]
            results.append(batch_embs)
            idx += len(batch)

        return {
            "model": request.model,
            "batches": results,
            "cache_stats": {
                "hits": cache_hits,
                "misses": cache_misses,
                "hit_rate": cache_hits / (cache_hits + cache_misses) if cache_hits + cache_misses > 0 else 0
            }
        }

    except Exception as e:
        logger.error(f"Batch embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Usage Example:**
```python
# Client-side: Index 1000 documents
import requests

documents = [f"Document {i} content..." for i in range(1000)]

# OLD WAY: 1000 separate requests
for doc in documents:
    response = requests.post("/api/embed", json={"model": "qwen25-1024d", "input": doc})

# NEW WAY: 1 batch request (10x faster!)
response = requests.post("/api/embed/batch", json={
    "model": "qwen25-1024d",
    "inputs": [documents],  # Single batch of 1000 docs
    "batch_size": 32
})
```

**RAM Impact:** +0MB (no additional memory, just code optimization)

**Expected Gains:**
- Network overhead: 1000 requests → 1 request
- Connection setup: 1000x → 1x
- Overall: 3-5x faster bulk operations

---

#### 1.3 Optimized Cosine Similarity

**Current Implementation (main.py:334-337):**
```python
# ❌ Slow: Python loop, not vectorized
scores = [
    np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
    for doc_emb in doc_embs
]
```

**Optimized Implementation:**
```python
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Fast: Vectorized BLAS operations
scores = cosine_similarity([query_emb], doc_embs)[0].tolist()
```

**Changes Required:**
```python
# main.py:328-337 (in rerank_documents)

# For Model2Vec models (replace lines 328-337)
if isinstance(selected_model, StaticModel):
    query_emb = selected_model.encode([request.query], show_progress_bar=False)[0]
    doc_embs = selected_model.encode(request.documents, show_progress_bar=False)

    # NEW: Use sklearn optimized version
    scores = cosine_similarity([query_emb], doc_embs)[0].tolist()
```

**RAM Impact:** +0MB (sklearn already installed via sentence-transformers)

**Expected Gains:**
- Reranking latency: 242ms → ~80-120ms (2-3x faster)
- More efficient for large document sets (>10 docs)

---

#### 1.4 Request Queue & Semaphore

**Purpose:** Protect against OOM during traffic spikes.

**Implementation:**
```python
import asyncio
from collections import deque
import time

class RequestQueue:
    """
    Request queue with concurrency limiting.

    Prevents OOM by limiting max concurrent requests.
    Queues excess requests instead of rejecting.
    """

    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue_size = 0
        self.max_queue = 100  # Reject if queue too long

    async def process(self, func, *args, **kwargs):
        if self.queue_size >= self.max_queue:
            raise HTTPException(
                status_code=503,
                detail=f"Server overloaded (queue: {self.queue_size})"
            )

        self.queue_size += 1
        try:
            async with self.semaphore:
                self.queue_size -= 1
                return await func(*args, **kwargs)
        except:
            self.queue_size -= 1
            raise

# Global queue (configurable via env)
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
request_queue = RequestQueue(max_concurrent=MAX_CONCURRENT)

@app.post("/api/embed")
async def create_embedding(request: EmbedRequest):
    # Wrap in queue
    return await request_queue.process(_do_embedding, request)

async def _do_embedding(request: EmbedRequest):
    """Actual embedding logic (moved from endpoint)"""
    # ... existing code
```

**Configuration:**
```bash
# .env
MAX_CONCURRENT_REQUESTS=10   # Max 10 concurrent embeddings
MAX_QUEUE_SIZE=100           # Reject if >100 queued
```

**RAM Impact:** +50MB (queue overhead, request tracking)

**Expected Gains:**
- Prevents OOM during traffic spikes
- Graceful degradation (queuing instead of crashing)
- Better error messages (503 instead of OOM crash)

---

#### 1.5 Metrics & Monitoring

**Purpose:** Know when to scale (data-driven decisions).

**Implementation:**
```python
from collections import deque
import time

class Metrics:
    """
    Track request metrics for scaling decisions.

    Monitors:
    - Concurrent users (1-minute window)
    - Average latency (last 100 requests)
    - Queue size
    - Cache hit rate
    """

    def __init__(self):
        self.requests = deque(maxlen=10000)  # Last 10k requests
        self.cache_hits = 0
        self.cache_misses = 0

    def record_request(self, latency_ms: float, cache_hit: bool = False):
        self.requests.append({
            "timestamp": time.time(),
            "latency": latency_ms
        })

        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def get_concurrent_users(self, window_sec: int = 60) -> int:
        """Estimate concurrent users (default: 1-min window)"""
        now = time.time()
        recent = [r for r in self.requests if now - r["timestamp"] < window_sec]
        return len(recent)

    def get_avg_latency(self, last_n: int = 100) -> float:
        """Average latency for last N requests"""
        if not self.requests:
            return 0.0
        recent = list(self.requests)[-last_n:]
        return sum(r["latency"] for r in recent) / len(recent)

    def get_cache_hit_rate(self) -> float:
        """Cache hit rate (percentage)"""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0

    def should_scale_to_phase2(self) -> bool:
        """Trigger for Phase 2 (Smart Concurrency)"""
        concurrent = self.get_concurrent_users(window_sec=60)
        avg_latency = self.get_avg_latency(last_n=100)

        # Scale if >10 concurrent users OR latency >100ms
        return concurrent > 10 or avg_latency > 100

    def should_scale_to_phase3(self) -> bool:
        """Trigger for Phase 3 (Multi-Worker)"""
        concurrent = self.get_concurrent_users(window_sec=60)
        avg_latency = self.get_avg_latency(last_n=100)

        # Scale if >50 concurrent users OR latency >200ms
        return concurrent > 50 or avg_latency > 200

# Global metrics
metrics = Metrics()

@app.get("/metrics")
async def get_metrics():
    """
    Internal monitoring endpoint.

    Use for:
    - Scaling decisions
    - Performance monitoring
    - Debugging
    """
    concurrent_1min = metrics.get_concurrent_users(window_sec=60)
    concurrent_5min = metrics.get_concurrent_users(window_sec=300)

    return {
        "status": "healthy",
        "timestamp": time.time(),

        # Traffic metrics
        "concurrent_users_1min": concurrent_1min,
        "concurrent_users_5min": concurrent_5min,
        "total_requests": len(metrics.requests),

        # Performance metrics
        "avg_latency_ms": round(metrics.get_avg_latency(last_n=100), 2),
        "p95_latency_ms": round(_calculate_percentile(metrics.requests, 95), 2),
        "p99_latency_ms": round(_calculate_percentile(metrics.requests, 99), 2),

        # Cache metrics
        "cache_hit_rate": round(metrics.get_cache_hit_rate(), 2),
        "cache_hits": metrics.cache_hits,
        "cache_misses": metrics.cache_misses,

        # Queue metrics
        "queue_size": request_queue.queue_size,
        "max_concurrent": request_queue.semaphore._value,

        # Scaling recommendations
        "scaling": {
            "current_phase": "Phase 1",
            "should_scale_to_phase2": metrics.should_scale_to_phase2(),
            "should_scale_to_phase3": metrics.should_scale_to_phase3(),
            "recommendation": _get_scaling_recommendation(metrics)
        }
    }

def _get_scaling_recommendation(metrics: Metrics) -> str:
    """Human-readable scaling recommendation"""
    if metrics.should_scale_to_phase3():
        return "⚠️ HIGH LOAD: Scale to Phase 3 (Multi-Worker) - >50 concurrent users"
    elif metrics.should_scale_to_phase2():
        return "⚠️ MODERATE LOAD: Consider Phase 2 (Smart Concurrency) - >10 concurrent users"
    else:
        return "✅ OPTIMAL: Current Phase 1 configuration sufficient"

def _calculate_percentile(requests: deque, percentile: int) -> float:
    """Calculate latency percentile"""
    if not requests:
        return 0.0
    latencies = sorted([r["latency"] for r in requests])
    idx = int(len(latencies) * percentile / 100)
    return latencies[min(idx, len(latencies) - 1)]
```

**Monitoring Dashboard:**
```bash
# Check metrics (internal endpoint)
curl http://localhost:11435/metrics | jq

{
  "status": "healthy",
  "concurrent_users_1min": 3,
  "concurrent_users_5min": 12,
  "avg_latency_ms": 2.34,
  "p95_latency_ms": 8.12,
  "p99_latency_ms": 15.45,
  "cache_hit_rate": 67.5,
  "queue_size": 0,
  "scaling": {
    "current_phase": "Phase 1",
    "should_scale_to_phase2": false,
    "should_scale_to_phase3": false,
    "recommendation": "✅ OPTIMAL: Current Phase 1 configuration sufficient"
  }
}
```

**RAM Impact:** +38MB (metrics storage for 10k requests)

**Expected Gains:**
- Data-driven scaling decisions (no guessing!)
- Early warning for performance degradation
- Clear metrics for optimization ROI

---

### Phase 1 Summary

**Total RAM Impact:**
```
LRU Cache:              +12MB
Batch endpoint:          +0MB
Cosine similarity:       +0MB
Request queue:          +50MB
Metrics:                +38MB
────────────────────────────
Total Phase 1:         +100MB
Current usage:           3GB
New usage:            3.1GB
════════════════════════════
Railway tier:    UNCHANGED ($15/month)
```

**Total Performance Impact:**
```
Latency (cache hits):   5ms → <0.5ms   (-90%)
Latency (cache misses): 5ms → 5ms      (unchanged)
Latency (weighted avg): 5ms → 2-3ms    (-40-60%)
Reranking:            242ms → 100ms    (-60%)
Bulk operations:       1000 req → 1 req (+300%)
OOM protection:        ❌ → ✅         (prevents crashes)
Monitoring:            ❌ → ✅         (scaling visibility)
```

**Implementation Checklist:**

- [ ] 1.1 LRU Cache
  - [ ] Add `encode_with_cache()` function
  - [ ] Add `EMBEDDING_CACHE_SIZE` env var
  - [ ] Add `EMBEDDING_MAX_CACHE_LENGTH` env var
  - [ ] Update `/api/embed` endpoint
  - [ ] Test cache hit/miss behavior

- [ ] 1.2 Batch Endpoint
  - [ ] Create `BatchEmbedRequest` model
  - [ ] Create `BatchEmbedResponse` model
  - [ ] Implement `/api/embed/batch` endpoint
  - [ ] Add cache stats to response
  - [ ] Write client example code
  - [ ] Load test (1000 docs single batch)

- [ ] 1.3 Cosine Similarity
  - [ ] Replace manual loop with sklearn
  - [ ] Update `/api/rerank` endpoint
  - [ ] Benchmark before/after

- [ ] 1.4 Request Queue
  - [ ] Create `RequestQueue` class
  - [ ] Add `MAX_CONCURRENT_REQUESTS` env var
  - [ ] Wrap endpoints in queue
  - [ ] Test queue overflow (503 errors)

- [ ] 1.5 Metrics
  - [ ] Create `Metrics` class
  - [ ] Add `/metrics` endpoint
  - [ ] Record request latency
  - [ ] Track cache hit rate
  - [ ] Test scaling triggers

- [ ] Testing & Validation
  - [ ] Load test with Apache Bench: `ab -n 1000 -c 10`
  - [ ] Verify RAM stays <3.2GB
  - [ ] Measure cache hit rate (target: >30%)
  - [ ] Check metrics endpoint accuracy

- [ ] Documentation
  - [ ] Update API docs with batch endpoint
  - [ ] Document cache configuration
  - [ ] Add monitoring guide
  - [ ] Update deployment instructions

---

## 🎯 Phase 2: Smart Concurrency (10-50 req/sec)

**Timeline:** 1 month after Phase 1
**Target Users:** 10-50 concurrent users
**Cost Impact:** +$5/month (~3.5GB total RAM)

### Triggers for Phase 2

**Monitoring signals:**
- ✅ `/metrics` shows `should_scale_to_phase2: true`
- ✅ Concurrent users >10 for >5 minutes
- ✅ Average latency >100ms
- ✅ Cache hit rate >30% (confirms caching works)

**Before implementing Phase 2:**
1. ✅ Verify Phase 1 cache hit rate >30%
2. ✅ Check if queue_size >5 regularly
3. ✅ Profile which endpoints are slow
4. ✅ Estimate actual concurrent user growth

### Objectives

1. ✅ Handle 10-50 concurrent users efficiently
2. ✅ Maintain <100ms average latency
3. ✅ Stay under 4GB RAM
4. ✅ Optimize for common query patterns

### Optimizations

#### 2.1 Request Coalescing (Deduplication)

**Purpose:** Avoid processing identical concurrent requests multiple times.

**Implementation:**
```python
from collections import defaultdict
import asyncio

# Global cache for pending requests
_pending_requests = defaultdict(lambda: {
    "future": None,
    "lock": asyncio.Lock()
})

async def deduplicated_encode(model_name: str, text: str):
    """
    Coalesce identical concurrent requests.

    If 10 users ask same query simultaneously:
    - OLD: Process 10x
    - NEW: Process 1x, share result with all 10
    """
    # Create unique key
    key = f"{model_name}:{hashlib.md5(text.encode()).hexdigest()}"

    async with _pending_requests[key]["lock"]:
        # Check if request already in progress
        if _pending_requests[key]["future"] is not None:
            logger.debug(f"Coalescing duplicate request: {key[:32]}...")
            # Wait for existing request to complete
            return await _pending_requests[key]["future"]

        # Create new request
        future = asyncio.create_task(_do_encode(model_name, text))
        _pending_requests[key]["future"] = future

        try:
            result = await future
            return result
        finally:
            # Clean up
            _pending_requests[key]["future"] = None

            # Optional: Clean old entries (prevent memory leak)
            if len(_pending_requests) > 10000:
                _cleanup_old_pending_requests()

async def _do_encode(model_name: str, text: str):
    """Actual encoding (with cache check)"""
    return encode_with_cache(model_name, [text])[0]
```

**RAM Impact:** +100MB (pending request tracking, worst case 10k concurrent)

**Expected Gains:**
- 30-50% reduction for duplicate concurrent queries
- Common in: FAQ bots, search autocomplete, real-time dashboards

---

#### 2.2 Preload Common Embeddings

**Purpose:** Eliminate first-request latency for common queries.

**Implementation:**
```python
# Load common queries from config file or database
COMMON_QUERIES_FILE = os.getenv("COMMON_QUERIES_FILE", "config/common_queries.json")

@app.on_event("startup")
async def preload_common_embeddings():
    """
    Preload embeddings for common queries at startup.

    Use cases:
    - FAQ systems (top 100 questions)
    - Search autocomplete (popular queries)
    - Dashboard queries (repeated every 5min)
    """
    if not os.path.exists(COMMON_QUERIES_FILE):
        logger.info("No common queries file found, skipping preload")
        return

    with open(COMMON_QUERIES_FILE) as f:
        common_queries = json.load(f)

    logger.info(f"Preloading {len(common_queries)} common query embeddings...")

    for query_data in common_queries:
        query = query_data["text"]
        model_name = query_data.get("model", "qwen25-1024d")

        if model_name in models:
            # Encode to populate cache
            _ = encode_with_cache(model_name, [query])

    logger.info("✅ Common embeddings preloaded!")

# config/common_queries.json
[
  {
    "text": "What is the status?",
    "model": "qwen25-1024d",
    "frequency": 150  // requests per day
  },
  {
    "text": "Explain how this works",
    "model": "qwen25-1024d",
    "frequency": 89
  },
  {
    "text": "Find documentation about API",
    "model": "qwen25-1024d",
    "frequency": 67
  }
  // ... up to 100 most common queries
]
```

**RAM Impact:** +400MB (100 queries × 4KB × 3 models + file overhead)

**Expected Gains:**
- 40-60% latency for common queries (5ms → <0.5ms on first request)
- Improved user experience (no "cold start" delay)

---

#### 2.3 AsyncIO Tuning

**Purpose:** Better handle I/O-bound operations.

**Configuration:**
```dockerfile
# Dockerfile - Add AsyncIO tuning
ENV UVICORN_BACKLOG=2048
ENV UVICORN_LIMIT_CONCURRENCY=100
ENV UVICORN_TIMEOUT_KEEP_ALIVE=5

CMD ["uvicorn", "src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "11435", \
     "--backlog", "2048", \
     "--limit-concurrency", "100", \
     "--timeout-keep-alive", "5"]
```

**RAM Impact:** +50MB (connection pool overhead)

**Expected Gains:**
- Better handling of connection bursts
- Faster request acceptance
- Reduced "connection refused" errors

---

### Phase 2 Summary

**Total RAM Impact:**
```
Phase 1 baseline:       3.1GB
Request coalescing:    +100MB
Preload queries:       +400MB
AsyncIO tuning:         +50MB
────────────────────────────
Total Phase 2:         3.65GB
════════════════════════════
Railway tier:    SAME or +1 tier (~$18-20/month)
```

**Total Performance Impact:**
```
Concurrent capacity:   1 user → 10-50 users
Duplicate query handling: +30-50% efficiency
Common query latency:    5ms → <0.5ms (preloaded)
Connection handling:     Better burst tolerance
```

**Implementation Checklist:**

- [ ] 2.1 Request Coalescing
  - [ ] Implement `deduplicated_encode()`
  - [ ] Add pending request tracking
  - [ ] Add cleanup mechanism (prevent leaks)
  - [ ] Test with concurrent identical requests

- [ ] 2.2 Preload Common Queries
  - [ ] Create `config/common_queries.json`
  - [ ] Implement preload at startup
  - [ ] Add logging for preload status
  - [ ] Monitor cache hit rate improvement

- [ ] 2.3 AsyncIO Tuning
  - [ ] Update Dockerfile with uvicorn flags
  - [ ] Test with connection burst (ab -c 100)
  - [ ] Monitor connection refused errors

- [ ] Testing & Validation
  - [ ] Load test: `ab -n 5000 -c 25`
  - [ ] Verify RAM <3.8GB
  - [ ] Test duplicate query coalescing
  - [ ] Measure preload impact

- [ ] Monitoring
  - [ ] Add coalescing metrics to `/metrics`
  - [ ] Track preload cache hit rate
  - [ ] Monitor Phase 3 triggers

---

## 🚢 Phase 3: Multi-Worker Scaling (>50 req/sec)

**Timeline:** 3 months after Phase 2
**Target Users:** 50-100+ concurrent users
**Cost Impact:** +$15-20/month (~6GB total RAM)

### Triggers for Phase 3

**Monitoring signals:**
- ✅ `/metrics` shows `should_scale_to_phase3: true`
- ✅ Concurrent users >50 for >10 minutes
- ✅ Average latency >200ms
- ✅ Queue size regularly >20

**Before implementing Phase 3:**
1. ✅ Verify Phase 1+2 optimizations working (cache hit >40%)
2. ✅ Profile CPU usage (should be >60% before multi-worker)
3. ✅ Estimate concurrent user growth trajectory
4. ✅ Test Railway RAM limits (can you use 6-8GB?)

### Objectives

1. ✅ Handle 50-100+ concurrent users
2. ✅ 2-4x throughput increase
3. ✅ Stay under 8GB RAM (target: 6GB)
4. ✅ Horizontal scaling ready

### Optimizations

#### 3.1 Multi-Worker with Lazy Loading

**Problem:** Each worker loads ALL models → RAM explosion.

**Solution:** Load different models per worker.

**Architecture:**
```
Worker 0 (PRIMARY - 3GB):
├── qwen25-1024d (65MB)   ← Fast, instruction-aware
├── gemma-768d (400MB)     ← Multilingual
├── qwen3-embed (600MB)    ← Full-size embeddings
├── embeddinggemma-300m    ← Full-size Gemma
└── qwen3-rerank          ← Reranking

Worker 1 (LIGHT - 500MB):
└── qwen25-1024d (65MB)    ← PRIMARY model only

Worker 2 (LIGHT - 500MB):
└── qwen25-1024d (65MB)    ← PRIMARY model only

════════════════════════════════
Total RAM: 3GB + 0.5GB + 0.5GB = 4GB
```

**Implementation:**
```python
# src/main.py - Modify startup

import os

@app.on_event("startup")
async def load_models():
    global models

    # Detect worker ID (Gunicorn sets WORKER_ID env var)
    worker_id = int(os.getenv("GUNICORN_WORKER_ID", "0"))

    logger.info(f"Starting worker {worker_id}...")

    # ALWAYS load primary model (qwen25-1024d)
    logger.info("Loading PRIMARY model: qwen25-1024d")
    models["qwen25-1024d"] = StaticModel.from_pretrained("tss-deposium/qwen25-deposium-1024d")
    logger.info("✅ qwen25-1024d loaded (65MB)")

    # Worker 0 loads ALL models (full-featured)
    if worker_id == 0:
        logger.info("Worker 0: Loading ALL models (full-featured)...")

        models["gemma-768d"] = StaticModel.from_pretrained("tss-deposium/gemma-deposium-768d")
        logger.info("✅ gemma-768d loaded (400MB)")

        models["qwen3-embed"] = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
        models["qwen3-rerank"] = models["qwen3-embed"]
        logger.info("✅ qwen3-embed + rerank loaded (600MB)")

        models["embeddinggemma-300m"] = SentenceTransformer("google/embeddinggemma-300m", trust_remote_code=True)
        logger.info("✅ embeddinggemma-300m loaded (300MB)")

        logger.info(f"Worker 0 ready: {len(models)} models loaded (~3GB)")
    else:
        # Workers 1-N: Only primary model (lightweight)
        logger.info(f"Worker {worker_id}: Lightweight mode (PRIMARY model only)")
        logger.info(f"Worker {worker_id} ready: 1 model loaded (~500MB)")

# Routing: Add header to route heavy models to Worker 0
@app.post("/api/embed")
async def create_embedding(request: EmbedRequest):
    # Check if model available in this worker
    if request.model not in models:
        # Model not loaded in this worker
        worker_id = int(os.getenv("GUNICORN_WORKER_ID", "0"))

        if worker_id != 0:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' only available on primary worker. Use 'qwen25-1024d' instead."
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not found"
            )

    # Process request
    return await request_queue.process(_do_embedding, request)
```

**Dockerfile changes:**
```dockerfile
# Install Gunicorn
RUN pip install gunicorn

# Multi-worker configuration
CMD ["gunicorn", "src.main:app", \
     "--workers", "3", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:11435", \
     "--timeout", "120", \
     "--preload"]
```

**RAM Impact:**
```
Worker 0 (full):        3GB
Worker 1 (light):     500MB
Worker 2 (light):     500MB
────────────────────────────
Total:                 4GB
════════════════════════════
vs Naive multi-worker: 9GB (3 × 3GB)
Savings:              -5GB (-56%)
```

**Expected Gains:**
- Throughput: 200 req/sec → 600 req/sec (3x for primary model)
- Concurrent capacity: 50 users → 150+ users
- Heavy model requests: Same speed (still worker 0)

---

#### 3.2 Redis Distributed Cache

**Purpose:** Share cache across workers.

**Problem:** Each worker has separate LRU cache → redundant computations.

**Solution:** Use Redis for shared cache.

**Implementation:**
```python
import redis
import pickle

# Connect to Redis (Railway can provision)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL, decode_responses=False)

def encode_with_redis_cache(model_name: str, texts: List[str]) -> List[np.ndarray]:
    """
    Encode with Redis distributed cache.

    Benefits:
    - Shared across all workers
    - Persistent across restarts
    - Configurable TTL
    """
    embeddings = []

    for text in texts:
        if len(text) > 500:
            # Long text: no cache
            emb = models[model_name].encode([text], show_progress_bar=False)[0]
            embeddings.append(emb)
            continue

        # Generate cache key
        cache_key = f"emb:{model_name}:{hashlib.md5(text.encode()).hexdigest()}"

        # Check Redis cache
        cached = redis_client.get(cache_key)

        if cached:
            # Cache hit
            emb = pickle.loads(cached)
            metrics.record_request(latency_ms=0.5, cache_hit=True)
        else:
            # Cache miss: compute and store
            emb = models[model_name].encode([text], show_progress_bar=False)[0]

            # Store in Redis (TTL: 1 hour)
            redis_client.setex(
                cache_key,
                3600,  # 1 hour TTL
                pickle.dumps(emb)
            )
            metrics.record_request(latency_ms=5, cache_hit=False)

        embeddings.append(emb)

    return embeddings
```

**Infrastructure:**
```bash
# Railway: Add Redis service
railway add redis

# Environment variables automatically set:
# REDIS_URL=redis://default:password@redis.railway.internal:6379
```

**RAM Impact:**
- Application: +50MB (Redis client overhead)
- Redis service: +512MB (separate Railway service)
- **Total: +562MB**

**Cost Impact:** +$5/month (Redis service on Railway)

**Expected Gains:**
- Cross-worker cache sharing
- Persistent cache (survives restarts)
- Higher cache hit rate (all workers contribute)

---

#### 3.3 Load Balancer Configuration

**Purpose:** Distribute requests efficiently across workers.

**Railway Configuration:**
```yaml
# railway.toml (if using Railway)
[deploy]
  numReplicas = 3

[health_check]
  path = "/health"
  timeout = 10
```

**Load Balancing Strategy:**
- Round-robin for `qwen25-1024d` requests (any worker)
- Sticky sessions for other models (force worker 0)

**Implementation:**
```python
# Add worker info to response headers
@app.middleware("http")
async def add_worker_header(request, call_next):
    response = await call_next(request)
    worker_id = os.getenv("GUNICORN_WORKER_ID", "0")
    response.headers["X-Worker-ID"] = worker_id
    return response
```

**RAM Impact:** +0MB (configuration only)

---

### Phase 3 Summary

**Total RAM Impact:**
```
Phase 2 baseline:       3.65GB
Multi-worker (3x):      +0.5GB (lazy loading saves 5GB!)
Redis cache:           +0.5GB
────────────────────────────
Total Phase 3:           4.65GB
════════════════════════════
Railway tier: ~6-8GB plan (~$30-35/month)
```

**Total Performance Impact:**
```
Concurrent capacity:   50 users → 150+ users (3x)
Throughput (qwen25):   200 → 600 req/sec (3x)
Throughput (heavy):    Same (still 1 worker)
Cache hit rate:        Higher (cross-worker sharing)
Resilience:            Better (worker failure tolerance)
```

**Implementation Checklist:**

- [ ] 3.1 Multi-Worker Setup
  - [ ] Add Gunicorn to requirements.txt
  - [ ] Implement worker ID detection
  - [ ] Add lazy loading logic
  - [ ] Update routing for heavy models
  - [ ] Test 3-worker configuration

- [ ] 3.2 Redis Cache
  - [ ] Provision Redis on Railway
  - [ ] Add redis-py to requirements
  - [ ] Implement distributed cache
  - [ ] Set appropriate TTL
  - [ ] Monitor Redis memory usage

- [ ] 3.3 Load Balancing
  - [ ] Configure Railway replicas
  - [ ] Add worker headers
  - [ ] Test request distribution
  - [ ] Verify sticky sessions (if needed)

- [ ] Testing & Validation
  - [ ] Load test: `ab -n 10000 -c 50`
  - [ ] Verify RAM <5GB
  - [ ] Test worker failure resilience
  - [ ] Measure cross-worker cache hit rate

- [ ] Monitoring
  - [ ] Add per-worker metrics
  - [ ] Track Redis cache hit rate
  - [ ] Monitor worker load distribution
  - [ ] Set up alerts for RAM/CPU

---

## 📊 Performance Benchmarks

### Expected Performance by Phase

| Metric | Baseline | Phase 1 | Phase 2 | Phase 3 |
|--------|----------|---------|---------|---------|
| **Single Request Latency** | 5ms | 2-3ms | 2-3ms | 2-3ms |
| **Cache Hit Latency** | N/A | <0.5ms | <0.5ms | <0.5ms |
| **Reranking Latency** | 242ms | 100ms | 100ms | 100ms |
| **Max Concurrent Users** | 1-5 | 5-10 | 10-50 | 50-150+ |
| **Throughput (req/sec)** | 200 | 200 | 500 | 600 |
| **Bulk Operations** | 1000 reqs | 1 req | 1 req | 1 req |
| **RAM Usage** | 3GB | 3.1GB | 3.65GB | 4.65GB |
| **Monthly Cost** | $15 | $15 | $20 | $35 |

### Benchmark Commands

```bash
# Phase 1 - Single threaded load test
ab -n 1000 -c 1 -p embed_request.json -T application/json \
   http://localhost:11435/api/embed

# Phase 2 - Moderate concurrency
ab -n 5000 -c 25 -p embed_request.json -T application/json \
   http://localhost:11435/api/embed

# Phase 3 - High concurrency
ab -n 10000 -c 50 -p embed_request.json -T application/json \
   http://localhost:11435/api/embed

# Batch endpoint test
ab -n 100 -c 10 -p batch_request.json -T application/json \
   http://localhost:11435/api/embed/batch
```

**Sample request files:**
```json
// embed_request.json
{
  "model": "qwen25-1024d",
  "input": "What is the status of the project?"
}

// batch_request.json
{
  "model": "qwen25-1024d",
  "inputs": [
    ["Doc 1", "Doc 2", "Doc 3"],
    ["Doc 4", "Doc 5", "Doc 6"]
  ],
  "batch_size": 32
}
```

---

## 🎛️ Configuration Reference

### Environment Variables

**Phase 1:**
```bash
# Cache configuration
EMBEDDING_CACHE_SIZE=1000           # Number of embeddings to cache
EMBEDDING_MAX_CACHE_LENGTH=500     # Only cache texts <500 chars

# Concurrency limits
MAX_CONCURRENT_REQUESTS=10         # Max concurrent embeddings
MAX_QUEUE_SIZE=100                 # Max queued requests

# Metrics
METRICS_RETENTION_SIZE=10000       # Keep last 10k requests
```

**Phase 2:**
```bash
# AsyncIO tuning
UVICORN_BACKLOG=2048
UVICORN_LIMIT_CONCURRENCY=100
UVICORN_TIMEOUT_KEEP_ALIVE=5

# Preload
COMMON_QUERIES_FILE=config/common_queries.json
```

**Phase 3:**
```bash
# Multi-worker
GUNICORN_WORKERS=3
GUNICORN_WORKER_CLASS=uvicorn.workers.UvicornWorker
GUNICORN_TIMEOUT=120

# Redis cache
REDIS_URL=redis://localhost:6379
REDIS_CACHE_TTL=3600              # 1 hour
```

### Railway Configuration

**Phase 1 (Starter plan):**
```yaml
# railway.toml
[build]
  builder = "dockerfile"

[deploy]
  startCommand = "uvicorn src.main:app --host 0.0.0.0 --port $PORT"
  healthcheckPath = "/health"
  restartPolicyType = "ON_FAILURE"
```

**Phase 3 (Pro plan with Redis):**
```yaml
# railway.toml
[build]
  builder = "dockerfile"

[deploy]
  numReplicas = 3
  startCommand = "gunicorn src.main:app --workers 3 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT"
  healthcheckPath = "/health"
  restartPolicyType = "ON_FAILURE"

[services]
  redis = { image = "redis:7-alpine", port = 6379 }
```

---

## 🚨 Scaling Decision Matrix

### When to Scale to Phase 2

| Metric | Threshold | Action |
|--------|-----------|--------|
| Concurrent users (1min) | >10 | Consider Phase 2 |
| Concurrent users (5min) | >15 | Implement Phase 2 |
| Avg latency | >100ms | Investigate + Phase 2 |
| Queue size | Regularly >5 | Phase 2 |
| Cache hit rate | <20% | Optimize queries first |

### When to Scale to Phase 3

| Metric | Threshold | Action |
|--------|-----------|--------|
| Concurrent users (1min) | >50 | Consider Phase 3 |
| Concurrent users (5min) | >75 | Implement Phase 3 |
| Avg latency | >200ms | Investigate + Phase 3 |
| Queue size | Regularly >20 | Phase 3 |
| CPU usage | >80% | Phase 3 (CPU-bound) |

### Pre-Scaling Checklist

**Before any scaling:**
1. ✅ Check `/metrics` endpoint for accurate data
2. ✅ Verify previous phase optimizations working
3. ✅ Profile slow endpoints (maybe 1 endpoint needs optimization)
4. ✅ Review cache hit rate (if <30%, optimize first)
5. ✅ Estimate cost increase vs value
6. ✅ Test in staging environment first

---

## 💰 Cost Analysis

### Total Cost of Ownership

| Phase | RAM | Railway Plan | Monthly Cost | Concurrent Users | Cost per 100 Users |
|-------|-----|--------------|--------------|------------------|-------------------|
| **Baseline** | 3GB | Starter | $15 | 1-5 | $300-1500 |
| **Phase 1** | 3.1GB | Starter | $15 | 5-10 | $150-300 |
| **Phase 2** | 3.65GB | Starter+ | $20 | 10-50 | $40-200 |
| **Phase 3** | 4.65GB | Pro | $35 | 50-150+ | $23-70 |

### ROI Calculation

**Phase 1 Investment:**
- Dev time: 1 week (~40 hours)
- Cost increase: $0/month
- Performance gain: -90% latency (cache hits), +300% bulk ops
- **ROI: INFINITE** (zero cost, massive gains)

**Phase 2 Investment:**
- Dev time: 1 week (~40 hours)
- Cost increase: +$5/month
- Performance gain: 10-50 concurrent users
- Cost per additional user: $5 / 40 users = **$0.125/user/month**

**Phase 3 Investment:**
- Dev time: 2 weeks (~80 hours)
- Cost increase: +$15/month (from Phase 2)
- Performance gain: 50-150 concurrent users
- Cost per additional user: $15 / 100 users = **$0.15/user/month**

**Conclusion:** Progressive scaling has excellent ROI. Phase 1 pays for itself immediately.

---

## 🔮 Future Considerations (Post-Phase 3)

### When Phase 3 Reaches Limits

**Scaling beyond 150 concurrent users:**

1. **Horizontal Scaling (Multiple Instances)**
   - Deploy multiple Railway services
   - Add external load balancer (Cloudflare, Nginx)
   - Estimated cost: +$35 per instance

2. **GPU Acceleration**
   - Move heavy models (qwen3-embed, rerank) to GPU
   - 10-50x speedup for full-size models
   - Estimated cost: +$50-100/month (GPU instance)

3. **Model Optimization**
   - Quantization (INT8, INT4) - 2-4x speedup, -75% RAM
   - ONNX conversion - 20-40% speedup
   - Model distillation - Create smaller models

4. **Microservices Architecture**
   - Separate services for each model
   - Independent scaling per model
   - More complex but ultimate scalability

5. **Edge Deployment**
   - CDN-based embedding cache
   - Regional model instances
   - Sub-10ms latency globally

### Technology Radar

**Monitor these technologies:**
- **Python 3.14 GIL-free** (6-12 months) - May enable better CPU concurrency
- **ONNX Runtime optimizations** - Continuous performance improvements
- **Model2Vec v2** - Next-gen distillation techniques
- **Hardware:** ARM-based servers (better performance/cost)

---

## 📚 References & Resources

### Internal Documentation

- **API Docs:** `/docs` endpoint (Swagger UI)
- **Monitoring:** `/metrics` endpoint
- **Health Check:** `/health` endpoint

### External Resources

**Model2Vec:**
- GitHub: https://github.com/MinishLab/model2vec
- Paper: "Distilling Sentence Embeddings from Large Language Models"

**Railway:**
- Docs: https://docs.railway.app
- Pricing: https://railway.app/pricing

**Performance Tools:**
- Apache Bench: https://httpd.apache.org/docs/current/programs/ab.html
- Locust (load testing): https://locust.io
- Grafana (monitoring): https://grafana.com

**Optimization Guides:**
- FastAPI Performance: https://fastapi.tiangolo.com/deployment/concepts/
- Uvicorn Tuning: https://www.uvicorn.org/settings/
- Gunicorn Best Practices: https://docs.gunicorn.org/en/stable/design.html

---

## ✅ Implementation Roadmap Timeline

### Month 1: Phase 1 Implementation

**Week 1:**
- [ ] Implement LRU cache
- [ ] Add batch endpoint
- [ ] Replace manual cosine similarity

**Week 2:**
- [ ] Add request queue/semaphore
- [ ] Implement metrics endpoint
- [ ] Load testing & validation

**Week 3:**
- [ ] Monitor production metrics
- [ ] Tune cache size based on hit rate
- [ ] Document learnings

**Week 4:**
- [ ] Buffer week for issues
- [ ] Plan Phase 2 if needed

### Month 2-3: Production Monitoring

- Monitor `/metrics` endpoint daily
- Track cache hit rate, latency, concurrent users
- Wait for Phase 2 triggers

### Month 4+: Phase 2/3 Based on Demand

- Implement only when metrics show need
- Progressive rollout
- Cost-benefit analysis at each step

---

## 🎯 Success Metrics

### Phase 1 Success Criteria

- [ ] Cache hit rate >30%
- [ ] Average latency <3ms (with cache)
- [ ] RAM usage <3.2GB
- [ ] Zero OOM crashes during load tests
- [ ] `/metrics` endpoint provides actionable data

### Phase 2 Success Criteria

- [ ] Handle 25 concurrent users with <100ms latency
- [ ] Request coalescing working (check logs)
- [ ] Preloaded queries <1ms latency
- [ ] RAM usage <3.8GB

### Phase 3 Success Criteria

- [ ] Handle 100 concurrent users with <150ms latency
- [ ] 3x throughput improvement
- [ ] Worker load balanced (check X-Worker-ID distribution)
- [ ] Redis cache working (cross-worker)
- [ ] RAM usage <5GB

---

## 📞 Support & Feedback

**Questions or issues during implementation?**

1. Check `/metrics` endpoint first
2. Review logs: `docker logs -f <container_id>`
3. Profile with `py-spy`: `py-spy top --pid <pid>`
4. Railway logs: `railway logs`

**Created:** 2025-10-20
**Last Updated:** 2025-10-20
**Version:** 1.0.0
**Status:** Ready for Phase 1 implementation
