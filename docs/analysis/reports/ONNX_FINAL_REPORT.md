# ONNX INT8 Optimization - Final Report

**Date**: 2025-10-12
**Status**: ONNX tested, insufficient speedup for Railway

---

## Executive Summary

ONNX INT8 quantization provides only **1.44x speedup** over PyTorch INT8, far below the target of 3-4x. This will **not solve** the Railway CPU performance issue (40s per paragraph).

---

## Benchmark Results

### Local Performance (WSL Ubuntu)
| Metric | PyTorch INT8 | ONNX INT8 | Improvement |
|--------|--------------|-----------|-------------|
| **Per-text latency** | 64.60ms | 44.84ms | **1.44x faster** |
| **Batch latency (5 texts)** | 323ms | 224ms | 1.44x faster |
| **Standard deviation** | ±149ms | ±34ms | More stable |

### Railway Production Estimate
With 2048 token context (2x longer) and Railway's 100x CPU throttling:
- **PyTorch INT8**: ~13 seconds per embedding
- **ONNX INT8**: ~9 seconds per embedding
- **Conclusion**: Still unusable on Railway

---

## Why ONNX Didn't Deliver

### Expected vs Actual
- **Target**: 3-4x speedup (10-15ms per embedding)
- **Actual**: 1.44x speedup (45ms per embedding)
- **Gap**: 208% slower than expected

### Root Causes
1. **PyTorch INT8 already optimized**: Dynamic quantization in PyTorch is quite efficient
2. **Architecture bottlenecks**: Gemma3 has inherent computational complexity that ONNX can't fully optimize
3. **CPU-bound operations**: Transformer attention is still expensive even with ONNX
4. **Railway throttling**: Even with 3-4x speedup, would still be ~3-5s per embedding on Railway (marginal)

---

## Path Forward: Two Options

### Option 1: GGUF Q4_K_M Quantization (Quick Test)
**Repository**: `sabafallah/embeddinggemma-300m-Q4_K_M-GGUF`

#### Pros
- ✅ **Quick to test** (can test today)
- ✅ **More aggressive quantization** (Q4 vs INT8 = smaller model, potentially faster)
- ✅ **GGUF optimized for CPU** (llama.cpp backend is highly optimized)
- ✅ **Easy to deploy** if it works
- ✅ **No training required**

#### Cons
- ⚠️ **More quality loss** (Q4 is 4-bit vs INT8's 8-bit)
- ⚠️ **Different infrastructure** (requires llama-cpp-python, not transformers)
- ⚠️ **Uncertain speedup** (might still be too slow on Railway)
- ⚠️ **GGUF compatibility** with existing API might need work

#### Expected Results
- **Best case**: 2-3x faster than ONNX INT8 (~15-20ms per embedding locally)
  - Railway estimate: ~2-3s per embedding (marginal but maybe acceptable)
- **Worst case**: Similar to ONNX INT8 (still too slow)
- **Quality**: Expect 2-5% MTEB loss vs baseline (Q4 is aggressive)

#### Technical Approach
1. Install `llama-cpp-python`
2. Download GGUF model
3. Create inference wrapper compatible with current API
4. Benchmark performance
5. Run MTEB to verify quality
6. **Timeline**: 1-2 days

---

### Option 2: Custom Model2Vec Distillation (High Risk, High Reward)
**Your earlier suggestion** - Distill gemma ourselves instead of using pre-trained TurboX.v2

#### Pros
- ✅ **Massive speedup** (100x+ like TurboX.v2: ~0.5ms per embedding)
- ✅ **Tiny model** (~30MB vs 300MB)
- ✅ **Railway viable** (0.5ms × 100x = 50ms per embedding on Railway ✅)
- ✅ **Control over quality** (can tune distillation dataset and parameters)
- ✅ **Better than TurboX.v2** (distilled from gemma, not Qwen3)

#### Cons
- ⚠️ **3-5 days of work** (dataset prep, training, evaluation)
- ⚠️ **Quality uncertain** (could be -10% to -70% MTEB loss)
- ⚠️ **Requires large dataset** (need representative corpus)
- ⚠️ **Requires MTEB validation** (can't deploy without quality check)
- ⚠️ **One-way decision** (if quality is bad, wasted time)

#### Expected Results
- **Best case**: 10-15% MTEB loss, 100x speedup (viable for Railway)
- **Realistic case**: 20-30% MTEB loss, 100x speedup (need to evaluate acceptability)
- **Worst case**: 50-70% MTEB loss (like TurboX.v2, unusable)

#### Key Insight
TurboX.v2 (0.233 MTEB, -70% loss) was distilled from **Qwen3**, not gemma. Distilling **from gemma** with a **high-quality dataset** might yield much better results.

#### Technical Approach
1. Prepare distillation dataset (1M+ diverse texts)
2. Use `model2vec` library to distill from gemma
3. Train with appropriate hyperparameters
4. Run full MTEB evaluation
5. If quality acceptable, deploy
6. **Timeline**: 3-5 days

---

## Recommendation: Test Both (Sequential)

### Phase 1: GGUF Q4_K_M (1-2 days)
**Rationale**: Quick win potential, low effort

1. Download and test `sabafallah/embeddinggemma-300m-Q4_K_M-GGUF`
2. Benchmark local performance
3. Run MTEB to check quality

**Decision Point**:
- ✅ If speedup is 2-3x vs ONNX AND quality loss <5%: Deploy to Railway and test
- ❌ If insufficient: Proceed to Phase 2

### Phase 2: Custom Model2Vec Distillation (3-5 days)
**Rationale**: Only option left for Railway CPU viability

1. Prepare high-quality distillation dataset
2. Distill gemma to Model2Vec
3. Run full MTEB evaluation
4. Deploy if quality is acceptable (>0.65 MTEB)

**Decision Point**:
- ✅ If MTEB >0.65: Deploy to Railway
- ❌ If MTEB <0.65: Consider Railway GPU or different base model

---

## Alternative: Railway GPU

If both GGUF and Model2Vec fail, consider:
- **Railway GPU pricing** for small GPU instance
- **Expected performance**: <5ms per embedding (100x faster than CPU)
- **No quality loss**
- **Cost**: Check Railway GPU tier pricing

---

## Files Created

### ONNX Implementation
- `venv_onnx/` - Isolated environment
- `download_onnx_model.py` - ONNX model downloader
- `test_onnx_model.py` - Model validation
- `benchmark_onnx.py` - Performance benchmarking
- `models/gemma-onnx-int8/` - Downloaded ONNX models (2.6GB)
- `onnx_benchmark.log` - Benchmark results

### Documentation
- `ONNX_SUCCESS.md` - Initial success report
- `ONNX_IMPLEMENTATION_STATUS.md` - Implementation tracking
- `ONNX_FINAL_REPORT.md` - This document

---

## Key Learnings

1. **ONNX is not a silver bullet**: 1.44x speedup insufficient for Railway
2. **PyTorch INT8 is already good**: Dynamic quantization works well
3. **Railway CPU severely throttled**: 100x slower than local (based on 40s paragraph issue)
4. **Need more aggressive optimization**: Q4 quantization or Model2Vec distillation
5. **Your Model2Vec insight was correct**: Custom distillation from gemma might outperform pre-trained TurboX.v2

---

## Next Steps

1. **Document ONNX findings** ✅ (this report)
2. **Test GGUF Q4_K_M** (1-2 days, low risk)
3. **If GGUF insufficient**: Custom Model2Vec distillation (3-5 days, high risk/reward)

---

**Last Updated**: 2025-10-12 21:05
**Decision**: Proceed with GGUF testing, then Model2Vec if needed
