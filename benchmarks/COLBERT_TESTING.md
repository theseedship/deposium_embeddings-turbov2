# 🔥 ColBERT Testing Guide

Guide pour tester et intégrer des modèles ColBERT (multi-vector embeddings) dans Deposium.

## 📖 Qu'est-ce que ColBERT ?

### Architecture

**ColBERT** = **Col**BERT: Efficient and Effective Passage Search via **Col**lection of **BERT** embeddings

### Différence clé vs Single-Vector

| Aspect | Single-Vector (Model2Vec, Sentence-Transformers) | Multi-Vector (ColBERT) |
|--------|--------------------------------------------------|------------------------|
| **Embeddings par texte** | 1 vecteur fixe (768D, 1024D, etc.) | N vecteurs (1 par token) |
| **Comparaison** | Cosine similarity directe | MaxSim operation |
| **Taille** | Petit (1 vector × dimensions) | Plus grand (N tokens × dimensions) |
| **Vitesse** | Très rapide | Plus lent (MaxSim sur tous tokens) |
| **Précision** | Bonne | Meilleure (contexte token-level) |

### MaxSim Operation

```python
# Single-Vector
similarity = cosine(query_embedding, doc_embedding)  # Simple dot product

# ColBERT
# Pour chaque token du query, trouver le max similarity avec tous les tokens du doc
# Puis moyenner ces max scores
similarities = []
for query_token_emb in query_embeddings:
    max_sim = max(cosine(query_token_emb, doc_token_emb) for doc_token_emb in doc_embeddings)
    similarities.append(max_sim)

overall_score = mean(similarities)
```

## 🔧 Installation

### Prérequis

```bash
pip install pylate torch
```

### Vérification

```python
from pylate import models, retrieve
print("✅ PyLate installé correctement")
```

## 📝 Template de Test

### Structure minimale

```python
from pylate import models, retrieve

# 1. Charger le modèle
model = models.ColBERT(
    model_name_or_path="mixedbread-ai/mxbai-edge-colbert-v0-32m"
)

# 2. Encoder query et documents
query = "How do I install Python?"
docs = [
    "Steps to set up Python on your computer",
    "Installing Java development kit",
    "Python package management with pip"
]

queries_embeddings = model.encode([query], is_query=True)
documents_embeddings = model.encode(docs, is_query=False)

# 3. Calculer scores avec MaxSim
scores = retrieve.score_maxsim(
    queries_embeddings=queries_embeddings,
    documents_embeddings=documents_embeddings,
)

# 4. Afficher résultats
for i, score in enumerate(scores[0]):
    print(f"Doc {i}: {score.item():.4f} - {docs[i]}")
```

## 📊 Benchmarks à Effectuer

### 1. Semantic Similarity

Tester sur paires similaires vs dissimilaires:

```python
similar_pairs = [
    ("The cat sat on the mat", "A feline rested on the rug"),
    ("ML is fascinating", "AI is interesting"),
]

dissimilar_pairs = [
    ("The cat sat on the mat", "Quantum physics is complex"),
]
```

**Métrique**: Séparation entre scores (similar avg - dissimilar avg)

### 2. Instruction Awareness

Comparer avec qwen25-1024d (94.9%):

```python
instruction_pairs = [
    ("How do I install Python?", "Steps to set up Python"),
    ("Can you explain recursion?", "Describe recursive functions"),
]
```

**Métrique**: Moyenne des scores d'instruction pairs

### 3. Code Understanding

Comparer avec qwen25-1024d (84.5%):

```python
code_pairs = [
    ("def hello(): print('hi')", "Function that prints hello"),
    ("for i in range(10): print(i)", "Loop printing 0 to 9"),
]
```

**Métrique**: Moyenne des scores code-description

### 4. Performance

```python
import time
import psutil

# RAM
process = psutil.Process()
mem_mb = process.memory_info().rss / 1024 / 1024

# Vitesse
start = time.time()
embeddings = model.encode(texts, is_query=True)
latency_ms = (time.time() - start) * 1000
```

**Métriques**:
- RAM usage (MB)
- Latency per text (ms)
- Model size (MB)

## 🎯 Critères de Décision

### ✅ Intégration Complète si:

1. **Qualité > qwen25-1024d**
   - Overall quality > 68.2%
   - Ou supérieur sur instruction-awareness OU code understanding

2. **Performance acceptable**
   - RAM overhead < 1GB
   - Latency < 100ms per query
   - Model size < 500MB

3. **Complexité acceptable**
   - Effort d'intégration < 2 jours
   - API compatible (peut être adapté)

### ⚠️ Test Only si:

- Qualité ≈ qwen25 (pas d'amélioration significative)
- Performance limitée (>100ms, >1GB RAM)
- Intégration complexe

### ❌ Reject si:

- Qualité < qwen25
- RAM > +2GB
- Latency > 200ms
- Incompatible avec architecture actuelle

## 🚀 Intégration Complète (Option B)

Si les tests sont positifs, voici l'effort requis:

### 1. Adapter FastAPI (src/main.py)

```python
# Ajouter endpoint ColBERT
@app.post("/api/colbert/embed")
async def colbert_embed(request: ColBERTEmbedRequest):
    """
    ColBERT embeddings (multi-vector)
    Returns: List of token embeddings per text
    """
    embeddings = colbert_model.encode(
        request.input,
        is_query=request.is_query
    )
    return {"embeddings": embeddings.tolist()}

@app.post("/api/colbert/score")
async def colbert_score(request: ColBERTScoreRequest):
    """
    Score query-document pairs with MaxSim
    """
    scores = retrieve.score_maxsim(
        queries_embeddings=query_embs,
        documents_embeddings=doc_embs
    )
    return {"scores": scores.tolist()}
```

### 2. Cache Strategy

ColBERT produit multi-vector → cache différent:

```python
# Cache key: hash(text + "query" or "doc")
cache = {}

def get_colbert_embedding(text, is_query):
    key = f"{hash(text)}_{is_query}"
    if key not in cache:
        cache[key] = model.encode([text], is_query=is_query)
    return cache[key]
```

### 3. N8N Integration

Node Ollama ne supporte pas multi-vector directement.

**Solutions**:
- Créer endpoint `/api/colbert/ollama-compat` qui retourne averaged embedding
- Ou créer custom N8N node pour ColBERT

## 📈 Comparaison avec Modèles Actuels

### vs qwen25-1024d (Production)

| Métrique | qwen25-1024d | ColBERT 32M | Diff |
|----------|--------------|-------------|------|
| Overall Quality | 68.2% | ? | ? |
| Instruction-Aware | 94.9% | ? | ? |
| Code Understanding | 84.5% | ? | ? |
| Model Size | 65MB | ~128MB | +63MB |
| RAM Total | 3.3GB | ? | ? |
| Latency | <1ms (cache) | ? | ? |
| Architecture | Single-vector | Multi-vector | - |

### vs gemma-768d (Backup)

| Métrique | gemma-768d | ColBERT 32M | Diff |
|----------|------------|-------------|------|
| Overall Quality | 65.9% | ? | ? |
| Multilingual | 69.0% | ? | ? |
| Model Size | 400MB | ~128MB | -272MB |
| Instruction-Aware | ❌ | ? | ? |

## 💡 Notes Importantes

### Pourquoi tester ColBERT ?

1. **Précision supérieure**: Late interaction = meilleur contexte
2. **Petit modèle**: 32M params = ~128MB (vs 600MB qwen3)
3. **Edge-optimized**: Conçu pour déploiement edge (Railway compatible)
4. **SOTA technique**: ColBERT = state-of-the-art pour retrieval

### Limitations connues

1. **Plus lent**: MaxSim operation vs simple cosine
2. **Plus complexe**: Multi-vector vs single-vector
3. **Cache différent**: Impossible de cacher single vector
4. **API différente**: N8N pas compatible direct

### Use Cases Potentiels

**✅ Bon pour**:
- RAG avec haute précision requise
- Code search (token-level matching)
- Question answering (instruction-aware)

**❌ Pas idéal pour**:
- Clustering (multi-vector incompatible)
- Classification simple
- Applications temps-réel strict (<10ms)

## 📚 Références

- **Blog Mixedbread**: https://www.mixedbread.com/blog/edge-v0
- **ColBERT Paper**: https://arxiv.org/abs/2004.12832
- **PyLate Docs**: https://github.com/lightonai/pylate
- **HuggingFace**: https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m

## 🔄 Workflow de Test

```bash
# 1. Installer
pip install pylate torch

# 2. Tester
cd benchmarks/models/mxbai-edge-colbert-32m
python test_colbert.py

# 3. Analyser
cat results.txt

# 4. Décider
# - Si positif → Plan intégration complète
# - Si neutre → Garder pour recherche future
# - Si négatif → Documenter et archiver
```

---

## ✅ Résultats de l'Évaluation

**Date**: 2025-10-18
**Modèle testé**: mxbai-edge-colbert-v0-32m
**Architecture**: ColBERT (multi-vector, late interaction)

### Scores Obtenus

| Métrique | Score | vs qwen25 |
|----------|-------|-----------|
| Overall Quality | **94.4%** | **+26.2%** |
| Instruction-Aware | **95.6%** | **+0.7%** |
| Code Understanding | **94.0%** | **+9.5%** |
| Semantic Similarity | **93.6%** | +39.6% |
| Multilingue FR | **91.97%** | -1.7% vs EN |
| Multilingue ES | **92.47%** | -1.1% vs EN |
| Multilingue DE | **90.26%** | -3.5% vs EN |

**Performance:**
- Model Size: 964 MB (vs 65MB qwen25)
- Encoding: 5.94 ms/text
- Context: 7999 tokens

### Décision Finale: ❌ **REJETÉ**

**Raison:** Overhead RAM prohibitif (+964MB, 15x plus gros que qwen25)

**Voir:** `benchmarks/models/mxbai-edge-colbert-32m/DECISION.md` pour l'analyse complète

**Conclusion:**
- ✅ Excellente qualité technique (meilleur modèle testé)
- ❌ Infrastructure cost trop élevé pour edge deployment
- 📚 Archivé comme référence "gold standard" (94.4%)
- 🎯 Confirme que qwen25-1024d est un excellent compromis qualité/ressources

**Statut**: ✅ Évaluation complète - Archivé pour référence