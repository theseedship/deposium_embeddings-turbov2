# Archive Index - ColBERT Evaluation

**Model**: mxbai-edge-colbert-v0-32m
**Date**: 2025-10-18
**Status**: ❌ Rejected for integration (RAM overhead)
**Quality**: 94.4% (best tested, but +964MB RAM)

---

## 📁 Structure des Fichiers

```
benchmarks/models/mxbai-edge-colbert-32m/
├── DECISION.md                    # 📋 Document de décision finale (À LIRE)
├── README.md                      # 📖 Spécifications du modèle
├── ARCHIVE_INDEX.md              # 📚 Ce fichier - Index des archives
├── results.txt                    # 📊 Résultats bruts des tests
│
├── test_colbert.py                # 🧪 Test principal (qualité + performance)
├── test_multilingual.py           # 🌍 Test multilingue (FR, ES, DE)
├── inspect_model.py               # 🔍 Inspection architecture
├── get_full_config.py             # ⚙️ Configuration complète
├── distill_to_model2vec.py        # 🔄 Tentative distillation (échec technique)
│
└── logs/
    ├── test_output.log            # 📝 Logs du test principal
    └── distillation.log           # 📝 Logs tentative distillation
```

---

## 📄 Description des Fichiers

### Documents de Décision

**DECISION.md** (7.9K) ⭐ **DOCUMENT PRINCIPAL**
- Analyse complète coût/bénéfice
- Raisons du rejet détaillées
- Comparaison avec qwen25-1024d
- Recommandation finale

**README.md** (4.1K)
- Spécifications techniques du modèle
- Architecture ColBERT expliquée
- Avantages/Inconvénients
- Statut de l'évaluation

**ARCHIVE_INDEX.md** (ce fichier)
- Index de tous les fichiers d'évaluation
- Guide de navigation dans l'archive

### Résultats

**results.txt** (1.4K)
- Résultats bruts des tests
- Scores de qualité (94.4%)
- Métriques de performance
- Recommandation automatique

### Scripts de Test

**test_colbert.py** (13K) ⭐ **TEST PRINCIPAL**
- Test de qualité globale
- Semantic similarity
- Instruction awareness
- Code understanding
- Performance metrics

**test_multilingual.py** (3.8K)
- Test support multilingue
- Français, Espagnol, Allemand
- Comparaison avec Anglais (baseline)
- Verdict par langue

**inspect_model.py** (3.9K)
- Inspection de l'architecture
- Dimensions embeddings
- Context length
- Configuration détaillée

**get_full_config.py** (2.2K)
- Extraction config complète HuggingFace
- Specs techniques détaillées
- Architecture ModernBERT

**distill_to_model2vec.py** (5.0K) ⚠️ **ÉCHEC TECHNIQUE**
- Tentative de distillation vers Model2Vec
- Objectif: réduire RAM overhead
- Résultat: Incompatibilité architecture multi-vector
- Conclusion: Distillation impossible

### Logs

**logs/test_output.log** (14K)
- Log complet du test principal
- Incluant tous les résultats détaillés
- Temps d'exécution: ~3 secondes

**logs/distillation.log** (9.1K)
- Log de la tentative de distillation
- Erreurs techniques (404 EntryNotFoundError)
- Preuve de l'incompatibilité architecture

---

## 🎯 Points Clés à Retenir

### ✅ Ce qui a fonctionné

1. **Tests de qualité excellents** : 94.4% overall (+26.2% vs qwen25)
2. **Instruction-awareness supérieur** : 95.6% (+0.7% vs qwen25)
3. **Code understanding excellent** : 94.0% (+9.5% vs qwen25)
4. **Support multilingue vérifié** : FR/ES/DE < 4% dégradation vs EN
5. **Performance acceptable** : 5.94 ms/text
6. **Méthodologie de test robuste** : Reproductible et documentée

### ❌ Ce qui a échoué

1. **Overhead RAM prohibitif** : +964MB (15x plus gros que qwen25: 65MB)
2. **Architecture incompatible** : Multi-vector vs single-vector
3. **Distillation impossible** : Model2Vec ne peut pas distiller multi-vector
4. **Wrapper "averaged" non viable** : Même RAM, perd late interaction
5. **Rapport qualité/RAM insuffisant** : 0.098% /MB vs qwen25: 1.05% /MB

### 📚 Leçons Apprises

1. **Multi-vector ≠ Single-vector** : Architectures fondamentalement différentes
2. **Edge-optimized ≠ Edge-deployable** : 32M params mais 964MB RAM
3. **Qualité excellente ≠ Intégration garantie** : Infrastructure cost matter
4. **ColBERT = Reference gold standard** : 94.4% est notre nouveau benchmark
5. **qwen25-1024d = Excellent compromise** : Validé par cette évaluation

---

## 🔗 Références

### Documentation Associée

- **`../../COLBERT_TESTING.md`** : Guide complet ColBERT (méthodologie)
- **`../../README.md`** : Tableau comparatif tous modèles (section ColBERT)

### Liens Externes

- **Blog Mixedbread** : https://www.mixedbread.com/blog/edge-v0
- **HuggingFace** : https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m
- **PyLate Library** : https://github.com/lightonai/pylate
- **ColBERT Paper** : https://arxiv.org/abs/2004.12832

---

## 📊 Comparaison Finale

| Métrique | qwen25-1024d (PROD) | ColBERT 32M (Rejeté) | Delta |
|----------|---------------------|----------------------|-------|
| Quality | 68.2% | **94.4%** | **+26.2%** ✅ |
| RAM | 65MB | 964MB | **+964MB** ❌ |
| Speed | <1ms (cache) | 5.94ms | +5ms ✅ |
| Instruction | 94.9% | 95.6% | +0.7% ✅ |
| Code | 84.5% | 94.0% | +9.5% ✅ |
| Multilingue | 39.4% | ~92% | +52.6% ✅ |
| Architecture | Single-vector | Multi-vector | Incompatible ❌ |
| **Quality/MB** | **1.05% /MB** | **0.098% /MB** | **-10.7x** ❌ |

**Verdict** : qwen25-1024d reste le meilleur choix pour edge deployment.

---

**Évaluation par** : Claude Code
**Validée par** : User (décision RAM overhead)
**Date** : 2025-10-18
**Statut** : ✅ Archive complète et documentée
