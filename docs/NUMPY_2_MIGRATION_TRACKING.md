# Numpy 2.0 Migration Tracking

**Status**: ⏳ Bloqué - En attente de mises à jour upstream
**Dernière révision**: 2025-12-21
**Version actuelle**: numpy==1.26.4

---

## Résumé

La migration vers numpy 2.0 est actuellement bloquée par des incompatibilités dans les dépendances ML critiques. Cette page documente les blockers et les ressources à surveiller.

---

## Blockers Actuels

### 1. PyTorch 2.6.0

| Aspect | Détail |
|--------|--------|
| **Version actuelle** | torch==2.6.0 |
| **Problème** | `torch.from_numpy()` incompatible avec numpy 2.x array protocol |
| **Issue GitHub** | https://github.com/pytorch/pytorch/issues/127522 |
| **Version cible** | PyTorch 2.7.0+ |
| **ETA** | Q1 2025 |

### 2. ONNX Runtime 1.23.2

| Aspect | Détail |
|--------|--------|
| **Version actuelle** | onnxruntime==1.23.2 |
| **Problème** | API numpy legacy utilisée internement |
| **Issue GitHub** | https://github.com/microsoft/onnxruntime/issues/19206 |
| **Version cible** | onnxruntime 1.24.0+ |
| **ETA** | Q1 2025 |

### 3. Dépendances transitives

```
deposium_embeddings-turbov2
├── torch==2.6.0 ❌ (requires numpy<2.0)
├── onnxruntime==1.23.2 ❌ (requires numpy<2.0)
├── optimum[onnxruntime]>=1.25.0
│   ├── torch ❌
│   └── onnxruntime ❌
├── transformers==4.51.0 ✅ (compatible)
└── sentence-transformers==5.1.2 ✅ (compatible)
```

---

## Ressources à Surveiller

### Releases

- **PyTorch**: https://github.com/pytorch/pytorch/releases
- **ONNX Runtime**: https://github.com/microsoft/onnxruntime/releases
- **Numpy**: https://github.com/numpy/numpy/releases

### Issues Clés

- PyTorch numpy 2.0 support: https://github.com/pytorch/pytorch/issues/127522
- ONNX Runtime numpy 2.0: https://github.com/microsoft/onnxruntime/issues/19206

---

## Checklist de Migration

Quand les blockers seront résolus:

- [ ] Vérifier la release PyTorch 2.7.0+
- [ ] Vérifier la release ONNX Runtime 1.24.0+
- [ ] Créer branche `feature/numpy-2-migration`
- [ ] Mettre à jour `requirements.txt`:
  ```diff
  - numpy==1.26.4
  + numpy>=2.0.0,<3.0.0
  ```
- [ ] Mettre à jour les dépendances ML:
  ```diff
  - torch==2.6.0
  + torch>=2.7.0
  - onnxruntime==1.23.2
  + onnxruntime>=1.24.0
  ```
- [ ] Exécuter les tests:
  ```bash
  pytest tests/ -v
  python -c "import numpy; print(numpy.__version__)"
  python -c "import torch; print(torch.from_numpy(numpy.array([1,2,3])))"
  ```
- [ ] Vérifier les performances (pas de régression)
- [ ] Tester sur GPU si applicable
- [ ] Merger dans main

---

## Pourquoi numpy 2.0 ?

### Avantages

1. **Performance**: Optimisations SIMD améliorées
2. **Mémoire**: Meilleure gestion des grands arrays
3. **API moderne**: Deprecations nettoyées
4. **Compatibilité future**: Base pour numpy 2.x+

### Risques

1. **API breaking changes**: Certaines fonctions deprecated supprimées
2. **Dépendances**: Toutes les libs ML doivent supporter numpy 2.0
3. **Tests**: Nécessite validation complète

---

## Historique

| Date | Action |
|------|--------|
| 2025-12-21 | Création du fichier de suivi, analyse des blockers |
| - | Prochaine révision: Quand PyTorch 2.7.0 sera annoncé |
