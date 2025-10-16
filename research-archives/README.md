# 🔬 Research Archives

This directory contains **experimental models and research results** that were tested but **NOT deployed to production**.

These models are archived here to:
- ✅ Preserve research results and benchmarks
- ✅ Enable future comparisons
- ✅ Document what was tested and why it wasn't adopted
- ⚠️ **NOT loaded automatically** by the API (to save RAM)

---

## 📁 Archived Models

### 1. Granite 4.0 Micro (200MB)
- **Path**: `granite-4.0-micro/granite-4.0-micro-deposium-1024d/`
- **Date**: 2025-10-16
- **Status**: ❌ **NOT RECOMMENDED FOR PRODUCTION**
- **Quality**: 84.08% (vs Qwen 93.46%)
- **Critical Issue**: 25% RAG accuracy in multilingual tests (unacceptable)

**Full Report**: See `granite-4.0-micro/GRANITE_FINAL_DECISION.md`

**Why archived**:
- ❌ -9.38% lower quality than Qwen2.5-1.5B
- ❌ Catastrophic multilingual RAG failure (25% accuracy)
- ❌ Inconsistent per-language performance (73% Spanish vs 99% Japanese)
- ✅ Only advantages: instruction-awareness (+2x), cross-lingual retrieval (+2x), smaller size (-34%)

**Strengths** (if needed for specialized use cases):
- ✅ Excellent French (94%), German (90%), Japanese (99.7%)
- ✅ 2x better instruction-awareness (15.21% vs 7.70%)
- ✅ 2x better cross-lingual retrieval (66.67% vs 33.33%)
- ✅ 34% smaller (200MB vs 302MB)

### 2. Qwen2.5-3B (302MB)
- **Path**: `qwen25-3b-deposium-1024d/`
- **Date**: 2025-10-14
- **Status**: ⚠️ **NOT TESTED IN PRODUCTION**
- **Quality**: 92.92% (similar to Qwen 1.5B: 93.46%)

**Why archived**:
- ⚠️ No significant advantage over Qwen2.5-1.5B
- ⚠️ Same size (302MB)
- ⚠️ Slightly slower (28,564 vs 33,527 texts/sec)

---

## 🎯 When to Use Archived Models

### Use Granite 4.0 Micro if:
- ✅ You need **French/German/Japanese** specifically (94-99% quality)
- ✅ You need **instruction-awareness** (15.21% vs 7.70%)
- ✅ You need **cross-lingual retrieval** (EN→FR, FR→EN)
- ❌ **BUT NOT for**: General-purpose, multilingual RAG, Spanish/Chinese

### Use Qwen2.5-3B if:
- ✅ You want to compare 1.5B vs 3B performance
- ✅ Research purposes only

---

## 📊 Comparison Summary

| Model | Quality | Multilingual | RAG | Speed | Size | Status |
|-------|---------|--------------|-----|-------|------|--------|
| **Qwen2.5-1.5B** | **93.46%** | **91.53%** | **100%** | **33,527 t/s** | 302MB | ✅ **PRODUCTION** |
| Qwen2.5-3B | 92.92% | 95.95% | 92.87% | 28,564 t/s | 302MB | ⚠️ Archived |
| Granite 4.0 Micro | 84.08% | 89.23% | 25% | 26,947 t/s | 200MB | ❌ Archived |
| Gemma-768D | 58.94% | 71.50% | 46.50% | 27,930 t/s | 383MB | ✅ **SECONDARY** |

---

## 🚀 Future Model Candidates

Models to test when available:
1. **Qwen2.5-7B** (expected 91-95% quality, better multilingual)
2. **Llama 3.2 3B** (Meta's latest small model)
3. **Phi-3.5-mini** (Microsoft's efficient model)
4. **Mistral-Small** (Mistral's compact model)

---

## 📝 Notes

- These models are **NOT loaded** by `src/main.py` automatically
- To test them, update `compare_all_models_v2.py` to point to `research-archives/`
- To deploy one, move it to `models/` and update `src/main.py`

---

**Last Updated**: 2025-10-16
**Research Duration**: ~3 hours (distillation + testing)
**Recommendation**: Continue using Qwen2.5-1.5B (v10.0.0) in production
