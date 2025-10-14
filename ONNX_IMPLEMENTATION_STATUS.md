# ONNX INT8 Implementation Status

**Date**: 2025-10-12
**Repo**: `deposium_embeddings-turbov2`
**Goal**: Optimize gemma-int8 for Railway CPU (currently 40s per paragraph - unusable)

---

## Current Situation

### Railway Performance Issue
- **gemma-int8 on Railway**: 40 seconds for a paragraph
- **Local benchmarks**: ~43ms per embedding (100x faster than Railway!)
- **Conclusion**: Railway CPUs are severely constrained/throttled

### MTEB Analysis Completed ✅
- **gemma-int8**: 0.788 average MTEB score (excellent quality)
- **TurboX.v2 (Model2Vec)**: 0.233 average MTEB score (-70% loss, catastrophic)
- **Decision**: Model2Vec pre-trained is not viable
- **Note**: Custom distillation from gemma might perform better (different from TurboX.v2)

---

## ONNX Conversion Progress

### Challenges Encountered
1. **Optimum Installation**: Dependency conflicts in main venv
2. **torch.onnx.export**: Gemma3 architecture incompatibility (`use_cache` argument issue)

### Solution: Fresh Virtual Environment
- **Location**: `venv_onnx/`
- **Status**: Installing dependencies (in progress)
- **Packages**: torch, transformers, onnxruntime, optimum[onnxruntime]
- **ETA**: 5-10 minutes

### Updated ONNX Conversion Approach
Fixed `convert_to_onnx.py` to use:
1. **Optimum export**: `ORTModelForFeatureExtraction.from_pretrained()` with `export=True`
2. **Save ONNX**: `model.save_pretrained()`
3. **Quantize manually**: `onnxruntime.quantization.quantize_dynamic()`
4. **Benefits**: Avoids torch.onnx.export issues, proper INT8 quantization

---

## Expected ONNX Results

### Target Metrics
| Metric | Current (gemma-int8) | Target (ONNX INT8) | Improvement |
|--------|----------------------|--------------------|-------------|
| **Latency (local)** | ~43ms/embedding | **~10-15ms/embedding** | **3-4x faster** |
| **Railway latency** | 40s paragraph (!) | **~4-12s paragraph** | **3-10x faster** |
| **MTEB Score** | 0.788 | **~0.77-0.775** | <2% loss |
| **Model Size** | ~300MB | **~200-250MB** | 20-30% smaller |

### Railway Viability
- If ONNX achieves 4-6s per paragraph: **Marginal** (slow but maybe acceptable)
- If ONNX achieves <4s per paragraph: **Production-ready** ✅
- If ONNX still >10s per paragraph: Need alternative strategy

---

## Alternative Strategies (If ONNX Fails)

### Option 1: Custom Model2Vec Distillation
**Rationale**: TurboX.v2 was distilled from Qwen3, not gemma. Distilling gemma ourselves with a high-quality dataset might perform much better.

**Pros**:
- Potential for 100x+ speedup (like TurboX.v2)
- ~30MB model size
- Control over distillation quality

**Cons**:
- Requires 3-5 days of work
- Quality uncertain (could be -10% to -70%)
- Need large representative dataset

**Next Steps**:
1. Test ONNX first
2. If ONNX insufficient, evaluate custom distillation
3. Run MTEB on distilled model before production

### Option 2: Different Base Model
**Examples**: `all-MiniLM-L6-v2`, `bge-small-en-v1.5`

**Pros**:
- Faster baseline (smaller models)
- Better ONNX support
- Proven Railway performance

**Cons**:
- Quality loss vs gemma
- Wasted investment in gemma

### Option 3: Railway GPU
**Cost**: Check Railway pricing for GPU instances

**Pros**:
- gemma-int8 with GPU would be <5ms per embedding
- No quality loss
- No optimization needed

**Cons**:
- Higher cost
- Requires Railway GPU quota

---

## Files Created

### MTEB Analysis
- `compare_models_mteb.py` - Benchmark script
- `PRELIMINARY_MTEB_ANALYSIS.md` - Early insights
- `FINAL_MTEB_DECISION.md` - Complete analysis
- `mteb_comparison.log` - Full results

### ONNX Conversion (In Progress)
- `convert_to_onnx.py` - Optimum-based conversion (fixed)
- `convert_to_onnx_alt.py` - torch.onnx alternative (failed)
- `test_onnx_model.py` - Testing script
- `benchmark_onnx.py` - Performance comparison
- `venv_onnx/` - Isolated environment (installing)
- `onnx_venv_install.log` - Installation log

---

## Next Steps (Immediate)

1. ⏳ **Wait for venv installation** (~5-10 min remaining)
2. **Run ONNX conversion**: `source venv_onnx/bin/activate && python3 convert_to_onnx.py`
3. **Test ONNX model**: `python3 test_onnx_model.py`
4. **Benchmark locally**: `python3 benchmark_onnx.py`
5. **Deploy to Railway**: Update Dockerfile, push
6. **Test on Railway**: Measure actual performance

### Success Criteria
- ✅ ONNX conversion completes without errors
- ✅ Local latency <15ms per embedding
- ✅ MTEB score >0.77 (<2% loss)
- ✅ Railway paragraph processing <5s

---

## Recommendation

**Proceed with ONNX INT8** as planned. If Railway performance is still >10s per paragraph after ONNX optimization, then evaluate:
1. Custom Model2Vec distillation (high risk, high reward)
2. Railway GPU upgrade (cost vs performance)
3. Alternative base model (quality tradeoff)

**Timeline**: ONNX conversion and testing should complete within 1-2 hours, giving us data to decide next steps.

---

**Last Updated**: 2025-10-12 20:50
**Status**: Installation in progress, conversion pending
