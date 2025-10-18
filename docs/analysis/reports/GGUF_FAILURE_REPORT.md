# GGUF Q4_K_M Test - Failure Report

**Date**: 2025-10-12
**Status**: GGUF architecture not supported by llama.cpp

---

## Executive Summary

GGUF Q4_K_M model is **not viable** because llama-cpp-python/llama.cpp does not support the `gemma-embedding` architecture. This is a newer embedding model architecture that hasn't been implemented in llama.cpp yet.

---

## What We Tried

### 1. Downloaded GGUF Model
- **Repository**: `sabafallah/embeddinggemma-300m-Q4_K_M-GGUF`
- **File**: `embeddinggemma-300m-q4_k_m.gguf`
- **Size**: 225.4 MB (Q4_K quantization)
- **Download**: ✅ Successful

### 2. Installed llama-cpp-python
- **Version**: 0.3.16
- **Installation**: ✅ Successful
- **Compilation**: ✅ Built with CPU support

### 3. Attempted Model Loading
- **Result**: ❌ Failed
- **Error**: `unknown model architecture: 'gemma-embedding'`

---

## Technical Details

### GGUF Model Metadata (From Verbose Output)

The model file is **valid GGUF V3** format with correct metadata:

```
Architecture:      gemma-embedding
Type:              model
Size:              300M
Context length:    2048
Embedding length:  768
Block count:       24
Feed forward:      1152
Attention heads:   3
Key/Value length:  256
Pooling type:      1 (mean pooling)
Quantization:      Q4_K - Medium (6.07 BPW)
```

### The Core Problem

```
llama_model_load: error loading model: error loading model architecture:
unknown model architecture: 'gemma-embedding'
```

**Root Cause**: The `gemma-embedding` architecture is not implemented in llama.cpp. llama.cpp currently supports:
- Standard LLM architectures (LLaMA, Mistral, Gemma for *text generation*)
- Some embedding models (e.g., BERT-based)
- **NOT** gemma-embedding (Google's embedding-specific architecture)

---

## Why GGUF Can't Work

1. **Architecture mismatch**: llama.cpp doesn't have support for gemma-embedding's unique architecture
2. **Not just quantization**: The issue is architectural support, not quantization format
3. **Upstream dependency**: Would require llama.cpp to add gemma-embedding support first
4. **No timeline**: Unknown when/if llama.cpp will add this support

---

## Alternative Paths

### Path 1: Wait for llama.cpp Support
- **Timeline**: Unknown (could be weeks/months)
- **Viability**: ❌ Not realistic for current timeline
- **Risk**: High - may never be implemented

### Path 2: Use Different GGUF Backend
- **Option**: Try different GGUF loader (e.g., ggml-python directly)
- **Viability**: ⚠️  Complex, uncertain success
- **Risk**: Medium-High - still needs gemma-embedding architecture support

### Path 3: Custom Model2Vec Distillation ✅ RECOMMENDED
- **Approach**: Distill gemma into Model2Vec format
- **Timeline**: 3-5 days
- **Expected speedup**: 100x+ (like TurboX.v2)
- **Viability**: ✅ High - proven technique
- **Risk**: Medium - quality uncertain but testable

---

## Comparison: ONNX vs GGUF vs Model2Vec

| Approach | Status | Local Speedup | Railway Estimate | Quality | Effort |
|----------|--------|---------------|------------------|---------|--------|
| **PyTorch INT8** | ✅ Baseline | 1.0x | ~13s per embedding | 0.788 MTEB | N/A |
| **ONNX INT8** | ✅ Tested | 1.44x | ~9s per embedding | Expected ~0.788 | 2 days |
| **GGUF Q4_K_M** | ❌ Not viable | N/A | N/A | N/A | Blocked |
| **Model2Vec** | ⏳ Next | **100x+** | **~50ms per embedding** | Unknown (0.55-0.75?) | 3-5 days |

---

## Decision: Proceed with Model2Vec

### Why Model2Vec is the Best Option

1. **Only viable path left**: ONNX insufficient, GGUF blocked
2. **Proven technique**: Model2Vec has 100x speedup track record
3. **Railway compatible**: 50ms per embedding is acceptable
4. **Better than TurboX.v2**: Distilling from gemma (not Qwen3) should improve quality
5. **No external dependencies**: Pure Python, works everywhere

### Model2Vec Approach

1. **Prepare distillation dataset** (1M+ diverse texts)
   - Use mix of domains: academic, web, code, conversational
   - Representative of Deposium use cases

2. **Distill gemma to Model2Vec** using model2vec library
   - Base model: `google/embeddinggemma-300m`
   - Target: Static embedding model (~30MB)

3. **Run full MTEB evaluation**
   - Target: >0.65 MTEB (vs 0.788 baseline)
   - TurboX.v2 was 0.233, custom distillation should be much better

4. **Deploy if quality acceptable**
   - If MTEB >0.65: Deploy to Railway ✅
   - If MTEB <0.65: Consider Railway GPU or different base model

---

## Key Learnings

1. **GGUF is not universal**: Not all model architectures supported
2. **llama.cpp is LLM-focused**: Primarily for text generation, not all embedding models
3. **Architecture matters**: Quantization format is not enough - need architectural support
4. **Upstream dependencies**: Can't use tools that don't support the architecture
5. **Model2Vec was the right insight**: Your original suggestion to distill from gemma is the viable path

---

## Files Created

### GGUF Attempt
- `download_gguf_model.py` - GGUF model downloader ✅
- `test_gguf_model.py` - Model validation script ✅
- `models/gemma-gguf-q4/` - Downloaded GGUF models ✅
- `gguf_download.log` - Download logs ✅
- `gguf_test.log` - Test results showing architecture error ✅
- `GGUF_FAILURE_REPORT.md` - This document ✅

---

## Next Steps

1. ✅ **Document GGUF failure** (this report)
2. ⏳ **Prepare Model2Vec distillation dataset**
3. ⏳ **Distill gemma to Model2Vec**
4. ⏳ **Run MTEB evaluation**
5. ⏳ **Deploy if quality acceptable**

---

## Timeline

- **ONNX Phase**: 2 days (complete) - 1.44x speedup insufficient
- **GGUF Phase**: 0.5 days (complete) - Architecture not supported ❌
- **Model2Vec Phase**: 3-5 days (next) - Expected 100x speedup, quality TBD

---

**Last Updated**: 2025-10-12 21:20
**Decision**: Proceed with custom Model2Vec distillation (only viable option remaining)
