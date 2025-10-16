# Guide d'Évaluation MTEB pour Qwen25-1024D

Guide complet pour évaluer Qwen25-1024D avec le benchmark MTEB officiel.

---

## 📊 Qu'est-ce que MTEB?

**MTEB (Massive Text Embedding Benchmark)** est le benchmark de référence pour évaluer les modèles d'embeddings.

### Couverture
- **58 datasets** couvrant 8 types de tâches
- **112 langues** supportées
- **~10M+ exemples** de test

### Types de tâches
1. **Classification** (7 datasets) - Banking77, Amazon Reviews, etc.
2. **Clustering** (11 datasets) - ArXiv, StackExchange, etc.
3. **Pair Classification** (3 datasets) - Duplicate questions
4. **Reranking** (4 datasets) - Question duplicates
5. **Retrieval** (15 datasets) - ArguAna, FiQA, NFCorpus, Quora, etc. ⭐ **PLUS IMPORTANT**
6. **STS** (17 datasets) - Semantic Textual Similarity
7. **Summarization** (1 dataset) - SummEval
8. **Bittext Mining** - Translation pair mining

---

## 🚀 Installation

### Option 1: Environnement virtuel (recommandé)

```bash
# Créer un venv dédié pour MTEB
python3 -m venv venv_mteb
source venv_mteb/bin/activate

# Installer les dépendances
pip install -r requirements_mteb.txt
```

### Option 2: Utiliser l'environnement Qwen25 existant

```bash
source venv_qwen25/bin/activate
pip install mteb datasets
```

---

## 📋 Modes d'Évaluation

### Mode 1: Quick Test (recommandé pour débuter)

**Durée:** ~30 minutes
**Tasks:** 7 tâches représentatives

```bash
python3 mteb_evaluation.py --mode quick
```

Teste:
- 1 Classification task
- 1 Clustering task
- 1 Pair Classification
- 2 Retrieval tasks (NFCorpus, SciFact)
- 2 STS tasks (STSBenchmark, SICK-R)

### Mode 2: Standard Test (balance temps/couverture)

**Durée:** ~2-3 heures
**Tasks:** ~20 tâches essentielles

```bash
python3 mteb_evaluation.py --mode custom --tasks \
  Banking77Classification \
  ArXivClusteringP2P \
  SprintDuplicateQuestions \
  AskUbuntuDupQuestions \
  ArguAna FiQA2018 NFCorpus QuoraRetrieval SCIDOCS SciFact TRECCOVID \
  STS12 STS13 STS14 STS15 STS16 STSBenchmark SICK-R \
  SummEval
```

### Mode 3: Full MTEB Benchmark (officiel)

**Durée:** ~4-8 heures (CPU) ou ~1-2 heures (GPU)
**Tasks:** 58 tâches complètes

```bash
python3 mteb_evaluation.py --mode full
```

---

## 🎯 Exécution

### Test Rapide (Quick Mode)

```bash
# Activer l'environnement
source venv_mteb/bin/activate

# Lancer le test rapide
python3 mteb_evaluation.py \
  --model models/qwen25-deposium-1024d \
  --output mteb_results_quick \
  --mode quick
```

### Test Complet

```bash
# Sur GPU (si disponible) - BEAUCOUP plus rapide!
CUDA_VISIBLE_DEVICES=0 python3 mteb_evaluation.py \
  --model models/qwen25-deposium-1024d \
  --output mteb_results_full \
  --mode full

# Sur CPU (plus lent mais fonctionne)
python3 mteb_evaluation.py \
  --model models/qwen25-deposium-1024d \
  --output mteb_results_full \
  --mode full
```

### Test Personnalisé

```bash
# Tester uniquement les tâches de Retrieval (les plus importantes)
python3 mteb_evaluation.py \
  --mode custom \
  --tasks ArguAna FiQA2018 NFCorpus QuoraRetrieval SCIDOCS SciFact TRECCOVID
```

---

## 📊 Résultats Attendus

### Comparaison avec Modèles de Référence

**Modèles Full-size (baseline):**
```
Model                          MTEB Score   Size      Speed
─────────────────────────────────────────────────────────────
text-embedding-3-large         64.59        ~1GB      Slow (API)
gte-large                      63.13        670MB     Medium
text-embedding-3-small         62.26        ~350MB    Medium
e5-large-v2                    62.25        1.34GB    Slow
instructor-xl                  61.79        4.96GB    Very Slow
text-embedding-ada-002         60.99        ~350MB    Medium (API)
```

**Qwen25-1024D (notre modèle):**
```
Model                          MTEB Score   Size      Speed
─────────────────────────────────────────────────────────────
Qwen25-1024D Model2Vec         ~45-55*      65MB      500-1000x FASTER!
```

*Note: Score estimé - Model2Vec sacrifie ~10-15 points MTEB pour gagner 500-1000x en vitesse*

### Pourquoi le score sera plus bas?

**Trade-offs de Model2Vec:**
- ❌ Score MTEB: ~45-55 (vs ~62-65 pour full-size)
- ✅ Taille: 65MB (vs 350MB-5GB)
- ✅ Vitesse: 500-1000x plus rapide
- ✅ Latence: <10ms (vs 50-500ms)
- ✅ Mémoire: ~100MB RAM (vs 1-8GB)
- ✅ Coût: Gratuit local (vs API payant)

### Où Qwen25-1024D excelle

**Forces attendues:**
1. **STS (Semantic Textual Similarity)** - Score élevé attendu (~60-70)
2. **Classification simple** - Très bon (~55-65)
3. **Clustering** - Bon (~50-60)

**Faiblesses attendues:**
1. **Retrieval complexe** - Plus faible (~40-50) mais acceptable
2. **Long documents** - Limité par le tokenizer

---

## 📁 Structure des Résultats

```
mteb_results/
├── qwen25-deposium-1024d_results.json          # Résultats complets JSON
├── Banking77Classification/
│   └── test_results.json                       # Résultats par task
├── STSBenchmark/
│   └── test_results.json
├── NFCorpus/
│   └── test_results.json
└── ...
```

### Format des résultats (JSON)

```json
{
  "Banking77Classification": {
    "test": {
      "accuracy": 0.8234,
      "f1": 0.8156,
      "main_score": 0.8234
    }
  },
  "STSBenchmark": {
    "test": {
      "cosine_pearson": 0.7856,
      "cosine_spearman": 0.7823,
      "main_score": 0.7823
    }
  }
}
```

---

## 🔬 Analyse des Résultats

### Commandes Utiles

```bash
# Voir le résumé
cat mteb_results_quick/qwen25-deposium-1024d_results.json | python3 -m json.tool | head -50

# Extraire le score moyen
python3 -c "
import json
with open('mteb_results_quick/qwen25-deposium-1024d_results.json') as f:
    data = json.load(f)
    scores = [v['test']['main_score'] for v in data.values() if 'test' in v]
    print(f'Average MTEB Score: {sum(scores)/len(scores):.4f}')
"

# Comparer avec baseline
python3 mteb_evaluation.py --compare mteb_results_full
```

---

## ⚡ Optimisation

### Pour GPU (Tesla T4, A100, etc.)

```bash
# Utiliser GPU
export CUDA_VISIBLE_DEVICES=0

# Augmenter batch size (Model2Vec est très rapide)
# Note: Model2Vec n'a pas de batch processing natif, mais MTEB le gère
python3 mteb_evaluation.py --mode full
```

### Pour CPU multi-core

```bash
# Utiliser tous les cores
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

python3 mteb_evaluation.py --mode full
```

---

## 📊 Publier les Résultats

### Sur HuggingFace Hub

```bash
# Upload results to model card
python3 -c "
from huggingface_hub import HfApi
api = HfApi()

api.upload_file(
    path_or_fileobj='mteb_results_full/qwen25-deposium-1024d_results.json',
    path_in_repo='mteb_results.json',
    repo_id='tss-deposium/qwen25-deposium-1024d',
    repo_type='model'
)
"
```

### Mettre à jour le README

Ajouter à `models/qwen25-deposium-1024d/README.md`:

```markdown
## MTEB Benchmark Results

| Task Type | Score | # Tasks |
|-----------|-------|---------|
| Classification | 0.XXX | 7 |
| Clustering | 0.XXX | 11 |
| PairClassification | 0.XXX | 3 |
| Reranking | 0.XXX | 4 |
| Retrieval | 0.XXX | 15 |
| STS | 0.XXX | 17 |
| Summarization | 0.XXX | 1 |
| **Overall** | **0.XXX** | **58** |

**Comparison:**
- Full-size models: ~62-65 MTEB score, 350MB-5GB, 50-500ms latency
- Qwen25-1024D: ~XX MTEB score, 65MB, <10ms latency (**500-1000x faster!**)
```

---

## 🐛 Troubleshooting

### Erreur: Out of Memory

```bash
# Réduire le nombre de tasks simultanées
# Lancer task par task
for task in Banking77Classification STSBenchmark NFCorpus; do
    python3 mteb_evaluation.py --mode custom --tasks $task
done
```

### Erreur: Dataset Download Failed

```bash
# Précharger les datasets
python3 -c "
from mteb import MTEB
tasks = MTEB(tasks=['Banking77Classification'])
tasks.run(None, output_folder='test', eval_splits=['test'])
"
```

### Trop Lent?

```bash
# Mode quick seulement (30 min au lieu de 4-8h)
python3 mteb_evaluation.py --mode quick

# Ou tester 1 seule task pour validation
python3 mteb_evaluation.py --mode custom --tasks STSBenchmark
```

---

## 📚 Ressources

- **MTEB Leaderboard:** https://huggingface.co/spaces/mteb/leaderboard
- **MTEB Paper:** https://arxiv.org/abs/2210.07316
- **MTEB GitHub:** https://github.com/embeddings-benchmark/mteb
- **Documentation:** https://github.com/embeddings-benchmark/mteb/tree/main/docs

---

## 🎯 Checklist d'Évaluation

- [ ] Environnement MTEB installé (`venv_mteb`)
- [ ] Modèle Qwen25-1024D disponible localement
- [ ] Test rapide (quick mode) exécuté (~30 min)
- [ ] Résultats quick analysés
- [ ] Test complet (full mode) lancé (~4-8h)
- [ ] Résultats publiés sur HuggingFace
- [ ] README mis à jour avec scores MTEB
- [ ] Comparaison avec baselines documentée

---

**Prêt pour l'évaluation MTEB! 🚀**

Commence avec `python3 mteb_evaluation.py --mode quick` pour un test rapide.
