# Model2Vec Distillation Strategy

**Date**: 2025-10-12
**Goal**: Maximize quality while achieving 100x speedup for Railway deployment

---

## Why Corpus-Based > Vocabulary-Based

### Vocabulary-Based (Simple, Lower Quality)
```
Input: Tokenizer vocabulary (~260k tokens)
Process: Embed each token individually
Result: Static lookup table
Problem: No context, no phrases, misses real-world patterns
```

### Corpus-Based (Better Quality)
```
Input: Diverse text corpus (1M+ sentences)
Process: Learn token embeddings from real usage patterns
Result: Context-aware static embeddings
Advantage: Captures how words are actually used together
```

---

## Optimal Dataset Composition

### For Deposium Use Case

Based on typical embedding use cases (semantic search, document similarity, retrieval):

#### 1. **Academic/Technical (30%)**
- Scientific papers abstracts
- Technical documentation
- Research articles
- **Why**: Deposium likely handles technical content

#### 2. **Web Content (30%)**
- Wikipedia articles
- News articles
- Blog posts
- **Why**: General knowledge representation

#### 3. **Conversational (20%)**
- Q&A pairs (StackOverflow, Reddit)
- Forum discussions
- Chat logs
- **Why**: Natural language queries

#### 4. **Code/Technical (10%)**
- Code comments
- API documentation
- Technical specifications
- **Why**: May embed technical content

#### 5. **Domain-Specific (10%)**
- Legal documents (if applicable)
- Business documents
- Specialized terminology
- **Why**: Deposium-specific use cases

---

## Data Sources (Practical)

### Option 1: Use Existing HuggingFace Datasets (Fast)

```python
from datasets import load_dataset

# 1. Wikipedia (high quality, diverse)
wiki = load_dataset("wikipedia", "20220301.en", split="train[:100000]")

# 2. Common Crawl (web content)
cc = load_dataset("c4", "en", split="train[:100000]", streaming=True)

# 3. Academic papers (arxiv)
arxiv = load_dataset("scientific_papers", "arxiv", split="train[:50000]")

# 4. Q&A (conversational)
qa = load_dataset("squad_v2", split="train[:50000]")

# 5. Code (technical)
code = load_dataset("code_search_net", "python", split="train[:20000]")
```

**Total**: ~320k diverse examples (good balance)

### Option 2: Use Sentence-Transformers Training Data (Optimal)

The `sentence-transformers` library has curated datasets specifically for embedding training:

```python
from sentence_transformers import datasets

# NLI (Natural Language Inference) - high quality
nli = datasets.NLIDataset("snli")

# STS (Semantic Textual Similarity) - high quality
sts = datasets.STSBenchmarkDataset()

# Paraphrases - captures semantic similarity
paraphrases = datasets.ParaphraseDataset()
```

---

## Model2Vec Distillation Parameters

### Key Parameters to Tune

```python
from model2vec import distill

model = distill(
    model=base_model,              # gemma-300m

    # Dataset approach (IMPORTANT!)
    vocabulary=None,                # Don't use vocabulary-only
    corpus=prepared_corpus,         # Use our curated corpus

    # Dimensionality
    pca_dims=768,                   # Keep original 768D (no compression)
    # OR pca_dims=512,              # Compress to 512D (faster, slight quality loss)

    # Vocabulary size (if using vocabulary)
    max_vocabulary_size=50000,      # Smaller = faster, larger = better coverage

    # Training parameters
    min_frequency=5,                # Ignore rare words
    window_size=5,                  # Context window for co-occurrence

    # Pooling strategy
    pooling="mean",                 # Mean pooling (same as gemma)
)
```

---

## Quality vs Speed Tradeoff

| Configuration | Quality (Est.) | Speed | Model Size | Railway Viable? |
|---------------|----------------|-------|------------|-----------------|
| **Vocabulary-only, 768D** | 0.45-0.55 MTEB | 100x | ~150MB | ❌ Too low quality |
| **Corpus-based, 768D** | 0.60-0.70 MTEB | 100x | ~200MB | ✅ Best balance |
| **Corpus-based, 512D** | 0.58-0.68 MTEB | 120x | ~130MB | ✅ Slightly faster |
| **Vocabulary-only, 512D** | 0.40-0.50 MTEB | 120x | ~100MB | ❌ Too low quality |

**Recommendation**: **Corpus-based, 768D** (best quality-speed tradeoff)

---

## Implementation Plan

### Phase 1: Dataset Preparation (2-3 hours)
1. Download diverse datasets from HuggingFace
2. Sample and balance across domains (30/30/20/10/10)
3. Clean and preprocess (remove duplicates, normalize)
4. Save as unified corpus (~1M sentences)

### Phase 2: Distillation (1-2 hours)
1. Load gemma-300m base model
2. Run Model2Vec distillation with corpus
3. Save distilled model

### Phase 3: Evaluation (1 hour)
1. Quick benchmark: Speed comparison
2. MTEB evaluation: Quality check
3. Decision: Deploy if MTEB >0.65

### Phase 4: Integration & Deployment (1-2 hours)
1. Update API to support Model2Vec
2. Test locally
3. Deploy to Railway
4. Verify performance

**Total Timeline**: 5-8 hours for full pipeline

---

## Success Criteria

### Minimum Viable
- ✅ MTEB score: >0.65 (vs 0.788 baseline = 17% loss)
- ✅ Speed: >50x faster than PyTorch INT8
- ✅ Railway latency: <1s per embedding

### Optimal
- ✅ MTEB score: >0.70 (vs 0.788 baseline = 11% loss)
- ✅ Speed: 100x+ faster
- ✅ Railway latency: <500ms per embedding

---

## Fallback Options

If quality is insufficient:

### Option A: Larger Vocabulary (Trade Speed for Quality)
- Increase vocabulary size from 50k → 100k
- Slightly slower (80x instead of 100x)
- Better coverage

### Option B: Hybrid Approach
- Use Model2Vec for common queries
- Fall back to gemma INT8 for complex queries
- Smart routing based on query complexity

### Option C: Fine-tune Model2Vec
- Use domain-specific corpus (Deposium data)
- Train on actual user queries
- Best quality for specific use case

### Option D: Railway GPU
- If CPU optimization insufficient
- Small GPU instance (~$10/month)
- <5ms per embedding, no quality loss

---

## Next Steps

1. ✅ Create dataset preparation script
2. ⏳ Download and prepare diverse corpus
3. ⏳ Run Model2Vec distillation with optimal parameters
4. ⏳ Benchmark speed and quality
5. ⏳ Deploy to Railway if viable

---

**Key Insight**: The quality of Model2Vec depends **critically** on the corpus used for distillation. Using a diverse, high-quality corpus (not just vocabulary) should yield much better results than pre-trained TurboX.v2.

**Expected Result**: 0.65-0.70 MTEB (vs TurboX.v2's 0.233) because we're distilling from **gemma** with a **curated corpus**, not from Qwen3 with unknown data.
