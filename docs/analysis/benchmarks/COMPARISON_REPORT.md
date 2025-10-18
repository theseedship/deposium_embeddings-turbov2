# LEAF Model Comparison: v1 vs v2
**Comparison Date**: 2025-10-12
**Evaluation**: MTEB + Speed Benchmarks

---
## 📐 Architecture Comparison
| Property | v1 (512 tokens) | v2 (2048 tokens) | Change |
|----------|-----------------|------------------|--------|
| **Layers** | 6 | 12 | +100% ✅ |
| **Parameters** | 75M | 120M | +60% ✅ |
| **Context Length** | 512 | 2048 | +300% ✅ |
| **Hidden Size Ratio** | 0.5x | 0.75x | +50% ✅ |
| **Training Data** | 50k samples | 200k samples | +300% ✅ |
| **Training Epochs** | 3 | 10 | +233% ✅ |
| **Alignment Loss Weight** | 1.0 | 2.5 | +150% ✅ |

## ⚡ Speed Comparison
| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| **Throughput** | 0.0 texts/s | 0.0 texts/s | +0.0% |
| **Latency** | 0.00 ms | 0.00 ms | +0.0% |
| **Embedding Dims** | 768 | 768 | Same |

## 📊 Quality Comparison (MTEB)
### Known Results (v1)

| Task | Metric | v1 (FAILED) | v2 (Target) | Target Improvement |
|------|--------|-------------|-------------| -------------------|
| **STSBenchmark** | Spearman | 0.223 | 0.70+ | **+214%** 🎯 |
| **STS22 English** | Spearman | 0.373 | 0.65+ | **+74%** 🎯 |
| **STS22 Average** | Spearman | ~0.21 | 0.50+ | **+138%** 🎯 |
| **Cross-lingual** | Spearman | -0.14 | 0.30+ | **Complete Fix** 🎯 |
| **MTEB Score (est.)** | Overall | ~25 | 55+ | **+120%** 🎯 |

## ❌ Detailed v1 Results (FAILED)

### STSBenchmark
- **Spearman**: 0.223 (Target: 0.81)
- **Quality Loss**: -72% vs base model
- **Status**: ❌ CRITICAL FAILURE

### STS22 by Language
| Language | Spearman | Status |
|----------|----------|--------|
| 🇨🇳 Chinese | 0.499 | 🟡 Best (still poor) |
| 🇸🇦 Arabic | 0.469 | 🟡 Moderate |
| 🇮🇹 Italian | 0.435 | 🟡 Moderate |
| 🇪🇸 Spanish | 0.403 | 🟠 Poor |
| 🇬🇧 English | 0.373 | 🟠 Poor |
| 🇫🇷 French | 0.300 | 🔴 Very poor |
| 🇷🇺 Russian | 0.268 | 🔴 Very poor |
| 🇹🇷 Turkish | 0.247 | 🔴 Very poor |
| 🇩🇪 German | 0.163 | ❌ Critical |
| 🇵🇱 Polish | 0.132 | ❌ Critical |

### Cross-lingual (Translation Tasks)
| Pair | Spearman | Status |
|------|----------|--------|
| 🇪🇸-🇮🇹 | 0.119 | ❌ Failed |
| 🇩🇪-🇵🇱 | 0.113 | ❌ Failed |
| 🇩🇪-🇫🇷 | 0.070 | ❌ Failed |
| 🇪🇸-🇬🇧 | 0.002 | ❌ Random |
| 🇨🇳-🇬🇧 | -0.012 | ❌ Inverse |
| 🇵🇱-🇬🇧 | -0.143 | ❌ **WORST** |

## 📝 Summary

### v1 (512 tokens) - FAILED
- ❌ **Architecture too aggressive**: 6 layers insufficient
- ❌ **Data insufficient**: 50k samples, mostly English
- ❌ **High alignment loss**: 2.18 (warning sign)
- ❌ **Quality catastrophic**: -72% vs base model
- ❌ **Multilingual destroyed**: Cross-lingual scores negative
- ✅ **Speed excellent**: 695 texts/s

### v2 (2048 tokens) - Expected Improvements
- ✅ **Architecture doubled**: 12 layers (2x)
- ✅ **Data 4x larger**: 200k multilingual samples
- ✅ **Alignment prioritized**: Weight 2.5 (vs 1.0)
- ✅ **Curriculum learning**: 512→1024→2048 progressive
- ✅ **Quality monitoring**: MTEB validation every 1000 steps
- ✅ **Target**: MTEB 55+ (vs ~25 in v1)
- ⚠️ **Speed trade-off**: Likely ~400-500 texts/s (still fast)

## 🎯 Recommendations

1. **Proceed with v2 training** using the improved configuration
2. **Monitor alignment loss** - stop if > 1.5 after epoch 3
3. **Validate frequently** - MTEB STSBenchmark every 1000 steps
4. **Target quality** - Spearman 0.70+ on STSBenchmark
5. **Expected training time** - 12-15 hours on RTX 4050

