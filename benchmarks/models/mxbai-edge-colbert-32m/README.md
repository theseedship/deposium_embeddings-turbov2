# mxbai-edge-colbert-v0-32m

## 📋 Informations du Modèle

- **Nom complet**: mixedbread-ai/mxbai-edge-colbert-v0-32m
- **Architecture**: ColBERT (Contextualized Late Interaction)
- **Paramètres**: 32 millions (32M)
- **Taille estimée**: ~128MB
- **Provider**: Mixedbread AI
- **Librairie**: PyLate (ColBERT implementation)

## 🎯 Particularités

### ColBERT = Multi-Vector Embeddings

**Différence clé avec nos modèles actuels**:
- **Modèles actuels** (qwen25, gemma): 1 vecteur par texte
- **ColBERT**: N vecteurs par texte (1 par token)

### Avantages

1. **Précision supérieure**: Late interaction capture mieux le contexte
2. **Petit modèle**: 32M params vs 600M+ pour alternatives
3. **Edge-optimized**: Conçu pour déploiement resource-constrained
4. **Token-level matching**: Meilleur pour code search, Q&A

### Inconvénients

1. **Plus lent**: MaxSim operation vs cosine similarity simple
2. **Plus complexe**: Architecture multi-vector
3. **Incompatible**: Avec clustering, certains benchmarks MTEB
4. **Cache différent**: Impossible de cacher single averaged vector

## 🧪 Tests Effectués

### 1. Semantic Similarity
Paires similaires vs dissimilaires pour mesurer séparation.

### 2. Instruction Awareness
Comparaison avec qwen25-1024d (94.9% baseline).

### 3. Code Understanding
Comparaison avec qwen25-1024d (84.5% baseline).

### 4. Performance
- RAM usage
- Encoding latency
- Model size

## 📊 Résultats

Voir `results.txt` pour les résultats détaillés.

**Comparaison avec qwen25-1024d (production)**:
- Overall Quality: ? vs 68.2%
- Instruction-Aware: ? vs 94.9%
- Code Understanding: ? vs 84.5%
- Model Size: ~128MB vs 65MB
- RAM Usage: ? vs 3.3GB total

## 🔧 Usage

### Test Simple

```bash
python test_colbert.py
```

### Programmatique

```python
from pylate import models, retrieve

# Charger modèle
model = models.ColBERT(
    model_name_or_path="mixedbread-ai/mxbai-edge-colbert-v0-32m"
)

# Encoder
query = "How do I install Python?"
docs = ["Steps to setup Python", "Installing Java"]

queries_emb = model.encode([query], is_query=True)
docs_emb = model.encode(docs, is_query=False)

# Score avec MaxSim
scores = retrieve.score_maxsim(
    queries_embeddings=queries_emb,
    documents_embeddings=docs_emb
)

print(scores)
```

## 🚀 Intégration Potentielle

### Si résultats positifs:

**Option A** (Simple):
- Endpoint API séparé `/api/colbert/embed`
- Utilisation pour cas spécifiques (code search, Q&A)
- Garde qwen25 pour usage général

**Option B** (Complète):
- Remplacer qwen25 si qualité >> meilleure
- Adapter cache et N8N
- Requiert refactoring significatif

### Si résultats neutres/négatifs:

- Documenter comme "testé mais pas retenu"
- Garder pour référence future
- Focus sur modèles single-vector

## 📚 Documentation

- **Guide ColBERT**: `../../COLBERT_TESTING.md`
- **Blog Mixedbread**: https://www.mixedbread.com/blog/edge-v0
- **HuggingFace**: https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m
- **PyLate**: https://github.com/lightonai/pylate

## 🔄 Statut

- [x] Installation pylate ✅
- [x] Test simple exécuté ✅
- [x] Test multilingue exécuté ✅
- [x] Résultats analysés ✅
- [x] **Décision prise: ❌ REJETÉ pour intégration** (RAM overhead)

**Voir [DECISION.md](./DECISION.md) pour l'analyse complète**

---

## 📊 Résultats Finaux

**Qualité:** 94.4% (meilleur modèle testé!)
**Instruction-Aware:** 95.6% (+0.7% vs qwen25)
**Code Understanding:** 94.0% (+9.5% vs qwen25)
**Multilingue:** FR/ES/DE excellent (< 4% dégradation)

**Verdict:** ❌ **Rejeté pour overhead RAM** (+964MB, 15x plus gros que qwen25)

Malgré l'excellente qualité (+26.2% vs qwen25), le modèle est **trop gourmand en RAM** pour notre use case edge deployment.

---

**Testé le**: 2025-10-18
**Architecture**: Multi-vector (ColBERT)
**Comparaison**: vs qwen25-1024d (single-vector)
**Statut**: Archivé comme référence "gold standard"