# 📊 Benchmarks - Comparaison des Modèles

Ce dossier contient tous les benchmarks et comparaisons des modèles d'embeddings testés pour Deposium.

## 🎯 Tableau Comparatif - Modèles Principaux

| Modèle | Dimensions | Taille | Qualité Globale | Instruction-Aware | Recommandation | Statut |
|--------|------------|--------|-----------------|-------------------|----------------|---------|
| **qwen25-1024d** ⭐ | 1024 | 65MB | **68.2%** | ✅ **94.9%** | ✅ Deploy | **PRODUCTION** |
| **gemma-768d** | 768 | 400MB | 65.9% | ❌ | ✅ Deploy | Backup |
| **qwen3-1024d** | 1024 | 600MB | 37.5% | ❌ | ❌ Do not deploy | Rejected |
| **qwen3-256d** | 256 | 100MB | 66.5% | ❌ | ⚠️ OK (limité) | Archive |
| **granite-4.0-micro** | - | - | ~86% (multilingual) | ❌ | ⚠️ Test only | Experimental |

### 🏆 Modèle Recommandé: **qwen25-1024d**

**Pourquoi ?**
- ✅ **Instruction-aware unique** (94.9%) - SEUL modèle avec cette capacité
- ✅ **Ultra compact** (65MB vs 400-600MB)
- ✅ **Qualité competitive** (68.2%)
- ✅ **Code understanding** (84.5%)
- ✅ **Conversational** (80.0%)
- ✅ **Multilingual** (39.4%)

## 📁 Structure des Résultats

```
benchmarks/
├── README.md                      # Ce fichier - Vue d'ensemble
├── model_comparison_results.json  # Comparaison globale
├── comparison_results.txt         # Résultats texte
│
├── models/                        # Résultats détaillés par modèle
│   ├── gemma-768d/
│   │   ├── results.json          # Scores détaillés
│   │   ├── eval_script.py        # Script d'évaluation
│   │   └── logs/                 # Logs d'exécution
│   ├── qwen25-1024d/
│   │   ├── results.json
│   │   ├── eval_script.py
│   │   └── distill_qwen25_1024d.py
│   ├── qwen25-7b/
│   ├── qwen3/
│   │   ├── qwen3_1024d_eval_results.json
│   │   ├── qwen3_256d_eval_results.json
│   │   └── qwen3_quick_eval_results.json
│   └── granite/
│       └── results.txt
│
├── comparisons/                   # Scripts de comparaison
│   ├── compare_all_models.py     # Compare tous les modèles
│   ├── compare_baseline_vs_qwen25.py
│   ├── compare_models_mteb.py
│   ├── compare_qwen25_vs_all.py
│   └── compare_versions.py
│
├── mteb/                         # Résultats MTEB (benchmarks standard)
│   ├── results/                  # Résultats complets
│   ├── results_baseline/         # Baseline de référence
│   └── results_quick/            # Tests rapides
│
└── tools/                        # Outils de benchmarking
    ├── benchmark.py
    ├── benchmark.sh
    ├── benchmark-simple.sh
    ├── benchmark_onnx.py
    ├── monitor_baseline.sh
    ├── monitor_granite.sh
    ├── monitor_mteb_live.sh
    └── extract_mteb_scores.sh
```

## 📈 Détails par Modèle

### 🔥 Qwen25-1024D (Production)

**Scores détaillés:**
- Semantic Similarity: 54.2%
- Topic Clustering: 43.4%
- Multilingual Alignment: 39.4%
- **Instruction Awareness: 94.9%** ⭐ (unique)
- **Conversational Understanding: 80.0%**
- **Code Understanding: 84.5%**

**Fichiers:**
- `models/qwen25-1024d/results.json`
- `models/qwen25-1024d/eval_script.py`
- `models/qwen25-1024d/distill_qwen25_1024d.py`

### ⚡ Gemma-768D (Backup)

**Scores détaillés:**
- Overall Quality: 65.9%
- Semantic Similarity: 73.0%
- Topic Clustering: 55.6%
- **Multilingual Alignment: 69.0%** (meilleur)
- Silhouette Score: 0.11
- Cluster Purity: 100%

**Fichiers:**
- `models/gemma-768d/results.json`
- `models/gemma-768d/eval_script.py`

### ❌ Qwen3-1024D (Rejected)

**Pourquoi rejeté:**
- Overall Quality: **37.5%** (trop faible)
- Semantic Similarity: 57.1%
- Topic Clustering: 35.0%
- Multilingual Alignment: **20.3%** (très faible)
- **-43.7% vs qwen3-256d** (régression massive)

**Fichiers:**
- `models/qwen3/qwen3_1024d_eval_results.json`

### 🧪 Granite 4.0 Micro (Experimental)

**Test multilingual:**
- English: 93.5%
- French: 94.0%
- German: 89.9%
- Spanish: 73.2%

**Fichiers:**
- `models/granite/results.txt`

## 🛠️ Comment Ajouter un Nouveau Modèle

1. **Créer le dossier:**
   ```bash
   mkdir -p benchmarks/models/nom-du-modele/logs
   ```

2. **Copier le script d'évaluation:**
   ```bash
   cp benchmarks/models/qwen25-1024d/eval_script.py benchmarks/models/nom-du-modele/
   # Adapter le script au nouveau modèle
   ```

3. **Lancer l'évaluation:**
   ```bash
   python benchmarks/models/nom-du-modele/eval_script.py > benchmarks/models/nom-du-modele/results.json
   ```

4. **Comparer avec les autres:**
   ```bash
   python benchmarks/comparisons/compare_all_models.py
   ```

5. **Mettre à jour ce README** avec les nouveaux résultats

## 📊 Scripts de Comparaison

### `compare_all_models.py`
Compare tous les modèles disponibles et génère un rapport complet.

### `compare_baseline_vs_qwen25.py`
Compare spécifiquement qwen25 avec la baseline de référence.

### `compare_models_mteb.py`
Compare les modèles en utilisant le benchmark MTEB standard.

### `compare_qwen25_vs_all.py`
Compare qwen25 avec tous les autres modèles (détaillé).

### `compare_versions.py`
Compare différentes versions d'un même modèle.

## 🎯 MTEB Benchmarks

Les résultats MTEB (Massive Text Embedding Benchmark) sont dans `mteb/`:

- **results/** - Évaluations complètes MTEB
- **results_baseline/** - Scores de référence (sentence-transformers)
- **results_quick/** - Tests rapides (sous-ensemble de tâches)

Documentation MTEB: `docs/guides/mteb/`

## 🔧 Outils de Monitoring

### Scripts de monitoring en temps réel:
- `tools/monitor_baseline.sh` - Monitor baseline evaluation
- `tools/monitor_granite.sh` - Monitor granite evaluation
- `tools/monitor_mteb_live.sh` - Monitor MTEB runs

### Extraction de scores:
- `tools/extract_mteb_scores.sh` - Extraire les scores MTEB des résultats

## 📚 Documentation Associée

- **Guides MTEB:** `docs/guides/mteb/`
  - MTEB_GUIDE.md
  - MTEB_QUICKSTART.md
  - MTEB_FINAL_ANALYSIS.md
  - MTEB_RESULTS_SUMMARY.md

- **Analyses benchmarks:** `docs/analysis/benchmarks/`
  - BENCHMARK_RESULTS.md
  - COMPARISON_REPORT.md

- **Analyses modèles:** `docs/analysis/models/`
  - MODEL_ANALYSIS.md
  - MODEL2VEC_STRATEGY.md

## 🚀 Prochaines Étapes

1. **Nouveau modèle à tester ?** Suivre la section "Comment Ajouter un Nouveau Modèle"
2. **Comparer les résultats ?** Utiliser `benchmarks/comparisons/compare_all_models.py`
3. **MTEB complet ?** Voir `docs/guides/mteb/MTEB_QUICKSTART.md`

## 💡 Notes

- **Instruction-awareness** est LA capacité unique de qwen25-1024d
- **Taille du modèle** est critique pour Railway (limites mémoire)
- **Multilingual** n'est pas prioritaire (focus français/anglais)
- **MTEB scores** donnent la vue d'ensemble sur tasks standards

---

📊 **Dernière mise à jour:** 2025-10-18
✨ **Modèle en production:** qwen25-1024d (65MB, instruction-aware)