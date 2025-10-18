# EmbeddingGemma Optimization Plan

## Context

**Problem**: EmbeddingGemma-300m float32 (~1.2GB) est trop lent sur Railway CPU:
- 512 tokens: ~15ms/embedding
- 1024 tokens: ~40ms/embedding
- 2048 tokens: ~115ms/embedding

**Solution**: Stratégie multi-variantes pour différents use cases

## Variantes Proposées

### 1. gemma (baseline) ✅ DEPLOYED
**Status**: Actuellement déployé
**Taille**: 587MB (float16) / 1.2GB (float32)
**Performance CPU**: ~115ms/2048 tokens
**Qualité**: MTEB 0.80 Spearman
**Use case**: Baseline de référence, GPU local

**Problème actuel**: Float16 CPU produit des zéros → Float32 fallback automatique

### 2. gemma-int8 (PRIORITÉ 1) 🎯
**Taille cible**: ~300MB (-75%)
**Performance CPU estimée**: ~40ms/2048 tokens (3x plus rapide)
**Qualité estimée**: MTEB 0.78-0.79 (-1-2%)
**Use case**: Production Railway CPU

**Implémentation**:
```python
from optimum.bettertransformer import BetterTransformer
from transformers import AutoModel
import torch

# Load with INT8 quantization
model = AutoModel.from_pretrained(
    "google/embeddinggemma-300m",
    load_in_8bit=True,  # INT8 quantization
    device_map="auto"
)
model = BetterTransformer.transform(model)  # Flash Attention optimizations
```

**Dépendances additionnelles**:
```
optimum>=1.16.0
bitsandbytes>=0.41.0
accelerate>=0.25.0
```

**Avantages**:
- ✅ Compatible sentence-transformers
- ✅ Pas de retraining nécessaire
- ✅ Perte de qualité minimale
- ✅ Rapide à déployer (~1 jour)

### 3. gemma-onnx-int8 (PRIORITÉ 2) ⚡
**Taille cible**: ~200MB (-83%)
**Performance CPU estimée**: ~25ms/2048 tokens (5x plus rapide)
**Qualité estimée**: MTEB 0.78-0.79 (-1-2%)
**Use case**: Production Railway CPU (max performance)

**Implémentation**:
```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
from optimum.onnxruntime.configuration import OptimizationConfig, QuantizationConfig

# Convert to ONNX with INT8 quantization
model = ORTModelForFeatureExtraction.from_pretrained(
    "google/embeddinggemma-300m",
    export=True
)

# Apply INT8 quantization
quantization_config = QuantizationConfig(is_static=False, format="QDQ")
model.quantize(save_dir="./gemma-onnx-int8", quantization_config=quantization_config)
```

**Dépendances additionnelles**:
```
onnxruntime>=1.17.0
optimum[onnxruntime]>=1.16.0
```

**Avantages**:
- ✅ Maximum performance CPU
- ✅ Taille optimale
- ✅ Compatible avec sentence-transformers via ORTModel
- ⚠️ Nécessite conversion (peut prendre quelques heures)

### 4. gemma-distilled (PRIORITÉ 3) 🚀
**Taille cible**: ~30MB (-97%)
**Performance CPU estimée**: ~2ms/2048 tokens (50x plus rapide)
**Qualité estimée**: MTEB 0.68-0.72 (-8-12%)
**Use case**: Applications temps réel, volume élevé

**Implémentation (Model2Vec)**:
```python
from model2vec import distill_model
from sentence_transformers import SentenceTransformer

# Load teacher model
teacher = SentenceTransformer("google/embeddinggemma-300m")

# Distill to static embeddings
distilled_model = distill_model(
    teacher,
    vocabulary_size=256000,  # Reduced vocabulary
    embedding_dim=768,       # Keep same dimensions
    output_folder="./gemma-distilled"
)
```

**Avantages**:
- ✅ Ultra rapide (pas de forward pass neural)
- ✅ Ultra compact
- ✅ Parfait pour embedding de vocabulaire fixe
- ⚠️ Perte de qualité significative pour contexte long

## Roadmap d'Implémentation

### Phase 1: INT8 Quantization (1-2 jours)
1. Ajouter dépendances optimum + bitsandbytes
2. Implémenter chargement INT8 avec load_in_8bit=True
3. Tester performance et qualité sur MTEB subset
4. Déployer sur Railway
5. Comparer latence vs gemma baseline

**Critères de succès**:
- Latence < 50ms pour 2048 tokens sur Railway CPU
- MTEB score > 0.78
- Taille < 350MB

### Phase 2: ONNX Conversion (2-3 jours)
1. Installer optimum[onnxruntime]
2. Convertir gemma-int8 en ONNX
3. Optimiser avec QuantizationConfig
4. Benchmark performance
5. Déployer variante gemma-onnx-int8

**Critères de succès**:
- Latence < 30ms pour 2048 tokens sur Railway CPU
- MTEB score > 0.78
- Taille < 250MB

### Phase 3: Model2Vec Distillation (3-5 jours)
1. Collecter dataset représentatif (1M textes)
2. Distiller avec Model2Vec
3. Évaluer sur MTEB
4. Déployer variante gemma-distilled

**Critères de succès**:
- Latence < 5ms pour 2048 tokens sur Railway CPU
- MTEB score > 0.68
- Taille < 50MB

## Configuration Multi-Variantes

### API Endpoints
```
POST /api/embed
{
  "model": "gemma",           // Baseline float16/float32
  "model": "gemma-int8",      // INT8 quantized (RECOMMANDÉ RAILWAY)
  "model": "gemma-onnx",      // ONNX INT8 optimized
  "model": "gemma-distilled", // Model2Vec ultra-fast
  "input": "text to embed"
}
```

### Selection Logic
```python
# Automatic model selection based on environment
if os.getenv("RAILWAY_ENVIRONMENT"):
    default_model = "gemma-int8"  # Railway CPU
elif torch.cuda.is_available():
    default_model = "gemma"        # Local GPU
else:
    default_model = "gemma-int8"  # Local CPU
```

## Benchmarks Attendus

| Variant | Size | Latency (2048 tok, CPU) | MTEB Score | Use Case |
|---------|------|------------------------|------------|----------|
| gemma (float32) | 1.2GB | ~115ms | 0.80 | Baseline, GPU local |
| gemma (float16) | 587MB | ⚠️ zeros bug | 0.80 | GPU local |
| **gemma-int8** | ~300MB | **~40ms** | **0.78-0.79** | **Railway CPU ✅** |
| gemma-onnx-int8 | ~200MB | **~25ms** | **0.78-0.79** | Railway CPU (optimal) |
| gemma-distilled | ~30MB | **~2ms** | 0.68-0.72 | High-volume, real-time |

## Next Steps

1. ✅ Déployer gemma baseline avec float16→float32 fallback (DONE)
2. 🎯 Implémenter gemma-int8 (START NOW)
3. ⏳ Convertir en ONNX INT8
4. ⏳ Distiller avec Model2Vec

## Questions à Résoudre

1. **Quality threshold**: Quel MTEB score minimum acceptable?
   - Pour production: 0.75+ recommandé
   - Pour high-volume: 0.65+ acceptable

2. **Multi-model strategy**: Charger toutes les variantes ou lazy loading?
   - Proposition: Charger seulement gemma-int8 sur Railway
   - Local: Charger gemma + gemma-int8 pour comparaison

3. **Fallback strategy**: Si gemma-int8 échoue, fallback vers?
   - Proposition: gemma-int8 → turbov2 (toujours disponible)

4. **Cost/Performance tradeoff**: Railway CPU pricing?
   - Latence actuelle: ~115ms → Risque de timeout
   - Latence cible: <50ms → Production-ready
