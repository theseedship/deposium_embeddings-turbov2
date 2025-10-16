# MTEB Evaluation - Quick Start

Évalue ton modèle Qwen25-1024D avec le benchmark MTEB officiel.

## 🚀 3 Options d'Évaluation

### Option 1: Quick Test (30 min) - RECOMMANDÉ POUR COMMENCER

```bash
./run_mteb_quick.sh
```

**7 tâches représentatives** pour avoir une première idée des performances.

### Option 2: Retrieval Only (1-2h) - POUR RAG/SEARCH

```bash
./run_mteb_retrieval.sh
```

**15 tâches de retrieval** - les plus importantes pour RAG, semantic search, Q&A.

### Option 3: Full Benchmark (4-8h) - OFFICIEL

```bash
./run_mteb_full.sh
```

**58 tâches complètes** - score MTEB officiel pour publication.

## 📊 Résultats Attendus

**Qwen25-1024D vs Full-size Models:**

| Modèle | MTEB Score | Taille | Latence | Use Case |
|--------|------------|--------|---------|----------|
| text-embedding-3-large | 64.59 | ~1GB | 50-200ms | Maximum qualité |
| gte-large | 63.13 | 670MB | 30-100ms | Haute qualité |
| **Qwen25-1024D** | **~45-55** | **65MB** | **<10ms** | **Speed + Efficiency** |

**Trade-off:** -10 à -15 points MTEB mais **500-1000x plus rapide!**

## 🎯 Pourquoi c'est important?

MTEB est le **benchmark de référence** pour les embeddings:
- Utilisé par OpenAI, Cohere, HuggingFace
- 58 datasets, 8 types de tâches
- Score comparable entre tous les modèles

## 📁 Résultats

Après l'évaluation:

```bash
# Voir le résumé
cat mteb_results_quick/qwen25-deposium-1024d_results.json | python3 -m json.tool | head -50

# Calculer le score moyen
python3 -c "
import json
data = json.load(open('mteb_results_quick/qwen25-deposium-1024d_results.json'))
scores = [v['test']['main_score'] for v in data.values() if 'test' in v]
print(f'Average MTEB Score: {sum(scores)/len(scores):.4f}')
"
```

## 📚 Guide Complet

Pour plus de détails: `MTEB_GUIDE.md`

---

**Recommandation:** Commence par `./run_mteb_quick.sh` pour avoir un premier score en 30 minutes! 🚀
