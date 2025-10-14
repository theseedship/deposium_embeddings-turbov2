# Preliminary MTEB Analysis: gemma-int8 vs TurboX.v2

**Status**: MTEB benchmarks in progress (awaiting complete results)
**Date**: 2025-10-12
**Purpose**: Evaluate if gemma should be distilled to Model2Vec format

---

## Early Insights (Based on Partial Data)

### ⏱️  Latency Results (CONFIRMED)

**Test**: Batch of 5 texts, 10 runs average

| Model | Avg Latency | Per Text | Relative Speed |
|-------|-------------|----------|----------------|
| **gemma-int8** (INT8 Quantized) | 213.29ms ± 84.72ms | ~42.66ms | Baseline (1.0x) |
| **TurboX.v2** (Model2Vec) | 1.71ms ± 0.48ms | ~0.34ms | **124.6x faster** |

**Analysis**:
- TurboX.v2 is **~125x faster** than gemma-int8
- gemma-int8 still averages ~43ms per embedding on CPU
- TurboX.v2 achieves sub-millisecond inference (~0.34ms per embedding)

**Railway Impact**:
- You reported gemma-int8 is "still too slow" on Railway CPU
- At ~43ms/embedding, batch operations would be slow
- TurboX.v2's <1ms latency is ideal for Railway CPU-only environment

---

### 🎯 Quality Results (PARTIAL - gemma-int8 only)

**MTEB Tasks Completed**:

#### BIOSSES (Biomedical Semantic Similarity)
- **gemma-int8**: 0.8584 Spearman
- TurboX.v2: *awaiting results*

#### STSBenchmark (Standard Semantic Textual Similarity)
- **gemma-int8**: 0.8262 Spearman
- TurboX.v2: *awaiting results*

#### STS17 (Multilingual, 11 subsets)
- *Currently evaluating...*

**Preliminary Average**:
- gemma-int8: ~0.842 Spearman (based on 2 tasks)
- TurboX.v2: *awaiting*

---

## Model Specifications

| Aspect | gemma-int8 | TurboX.v2 |
|--------|------------|-----------|
| **Type** | INT8 Quantized Transformer | Static Model2Vec Embeddings |
| **Size** | ~300MB | ~30MB |
| **Dimensions** | 768D | 1024D |
| **Max Context** | 2048 tokens | Token-based (static vocabulary) |
| **Architecture** | EmbeddingGemma-300m quantized | Qwen3 distilled to static |
| **Inference** | Neural forward pass (CPU) | Dictionary lookup (ultra-fast) |

---

## Current Status

### What We Know:
1. **Speed**: TurboX.v2 is **125x faster** ✅
2. **Size**: TurboX.v2 is **10x smaller** ✅
3. **Quality (gemma-int8)**: Strong performance (~0.842 on tested tasks) ✅
4. **Quality (TurboX.v2)**: *Awaiting MTEB results* ⏳

### What We're Waiting For:
1. TurboX.v2 MTEB scores on all 3 tasks
2. Direct quality comparison (gemma-int8 vs TurboX.v2)
3. Final recommendation from comparison script

---

## Preliminary Observations

### Speed Advantage (TurboX.v2)
The 125x speedup is **massive**:
- gemma-int8: 213ms for 5 texts = ~43ms each
- TurboX.v2: 1.71ms for 5 texts = ~0.34ms each

**For Railway CPU production**:
- Processing 1000 embeddings:
  - gemma-int8: ~43 seconds
  - TurboX.v2: ~0.34 seconds (125x faster)

This aligns with your observation that "gemma-int8 is still too slow on Railway with CPU".

### Quality Trade-off (Pending)

**Key Question**: Is the quality loss acceptable?

Expected scenarios based on Model2Vec characteristics:

1. **Best Case** (Quality Loss <5%):
   - TurboX.v2: 0.79-0.80+ Spearman
   - **Decision**: Use TurboX.v2 (faster + comparable quality)

2. **Moderate Case** (Quality Loss 5-10%):
   - TurboX.v2: 0.75-0.79 Spearman
   - **Decision**: Use TurboX.v2 for high-volume, gemma-int8 for high-quality tasks

3. **High Loss** (Quality Loss >10%):
   - TurboX.v2: <0.75 Spearman
   - **Decision**: Keep gemma-int8, consider ONNX optimization instead

---

## Potential Outcomes & Recommendations

### Scenario A: TurboX.v2 Quality ≥ 0.78 (Loss <8%)
**Recommendation**: ✅ **Switch to TurboX.v2 for Railway production**

**Rationale**:
- 125x faster (critical for Railway CPU)
- 10x smaller (reduced memory/bandwidth)
- Quality loss acceptable for most use cases
- Already available (no distillation needed!)

**Action Items**:
- Update `src/main.py` to set TurboX.v2 as default for Railway
- Keep gemma-int8 as optional for high-quality requirements
- Update OPTIMIZATION_PLAN.md to deprioritize Model2Vec distillation

### Scenario B: TurboX.v2 Quality 0.70-0.78 (Loss 8-17%)
**Recommendation**: ⚖️  **Hybrid Strategy**

**Rationale**:
- TurboX.v2 for high-volume/real-time (acceptable quality loss)
- gemma-int8 for high-quality requirements
- Use case determines model selection

**Action Items**:
- Implement smart model selection based on use case
- Add endpoint parameter: `quality_mode: "fast" | "balanced" | "high"`
- Document trade-offs for users

### Scenario C: TurboX.v2 Quality <0.70 (Loss >17%)
**Recommendation**: ❌ **Keep gemma-int8, pursue ONNX optimization**

**Rationale**:
- Quality loss too significant
- ONNX INT8 could provide 3-5x speedup with minimal quality loss
- Model2Vec distillation would require significant effort

**Action Items**:
- Focus on ONNX conversion (OPTIMIZATION_PLAN Phase 2)
- Target: ~30-50ms latency with <2% quality loss
- Evaluate if that's acceptable for Railway

---

## Next Steps

### Immediate (Current Session):
1. ⏳ Wait for MTEB benchmark completion (~5-10 more minutes)
2. ✅ Review full comparison report
3. ✅ Analyze quality vs speed trade-off
4. ✅ Make informed decision on Model2Vec distillation

### Post-Benchmark:
1. Update OPTIMIZATION_PLAN.md based on results
2. Implement recommended strategy
3. Update Railway deployment configuration
4. Document model selection guidelines

---

## Technical Notes

### Why TurboX.v2 is So Fast:
- **Model2Vec architecture**: Static embeddings (pre-computed word vectors)
- **No neural computation**: Dictionary lookup instead of forward pass
- **CPU-optimized**: No matrix operations, just vector retrieval
- **Memory efficient**: Smaller footprint, better cache utilization

### Why gemma-int8 is Slower:
- **Neural forward pass**: Multiple transformer layers
- **INT8 operations**: Still requires computation (even if quantized)
- **CPU limitation**: No GPU acceleration on Railway
- **Batch overhead**: Even small batches require full model forward pass

---

## Open Questions

1. **TurboX.v2 MTEB Score**: What's the actual quality on STS tasks?
2. **Quality-Speed Trade-off**: Is 125x speedup worth X% quality loss?
3. **Use Case Sensitivity**: Which Deposium features require high vs fast embeddings?
4. **Model2Vec Distillation**: Is it still worth pursuing given TurboX.v2 exists?

---

**Last Updated**: 2025-10-12 19:50
**Full Results Expected**: ~2025-10-12 20:00
**Comparison Script**: `compare_models_mteb.py`
**Log File**: `mteb_comparison.log`
