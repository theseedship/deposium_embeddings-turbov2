# Distillation Qwen2.5-7B → Model2Vec - Guide Complet

**🎯 Objectif:** Convertir Qwen/Qwen2.5-7B-Instruct (14GB) en Model2Vec 1024D (~65MB)
**📊 Performance cible:** 91-95% (+7-11% vs baseline Qwen2.5-1.5B)
**⚡ Priorité:** ABSOLUE
**📅 Date:** 2025-10-14

---

## 📦 Contenu du Projet (11 fichiers)

### Scripts Python (3)
1. **distill_qwen25_7b.py** - Script principal de distillation
2. **test_qwen25_7b_model.py** - Tests de validation du modèle
3. **quick_eval_qwen25_7b_1024d.py** - Évaluation complète (6 catégories)

### Scripts Shell (5)
4. **setup_qwen25_7b_env.sh** - Configuration de l'environnement
5. **run_qwen25_7b_distillation.sh** - Pipeline automatisé de distillation
6. **test_qwen25_7b_model.sh** - Exécution automatique des tests
7. **evaluate_qwen25_7b.sh** - Exécution automatique de l'évaluation
8. **deploy_qwen25_7b.sh** - Déploiement en production

### Documentation (3)
9. **QWEN25_7B_README.md** - Vue d'ensemble (EN)
10. **QWEN25_7B_QUICKSTART.md** - Démarrage rapide (EN)
11. **QWEN25_7B_DISTILLATION_GUIDE.md** - Guide complet (EN)
12. **PRE_DISTILLATION_CHECKLIST.md** - Checklist pré-distillation
13. **LISEZMOI_QWEN25_7B.md** - Ce fichier (FR)

---

## ⚙️ Votre Configuration Matérielle

**Détecté sur votre machine:**
```
✅ CPU:       OK (suffisant)
⚠️  RAM:       19GB (recommandé: 32GB+)
✅ Disque:    852GB (excellent!)
⚠️  GPU:       RTX 4050 5GB (recommandé: 16GB+)
```

**Impact:**
- ⚠️ GPU insuffisant → Risque de Out Of Memory (OOM)
- ✅ **Solution:** Mode CPU (plus lent mais stable)
- ⏱️ **Temps:** 10-20 heures au lieu de 2-4h

---

## 🚀 Démarrage Rapide (Pour Votre Machine)

### Étape 1: Configuration (5 min)
```bash
./setup_qwen25_7b_env.sh
```

Ce script va:
- Créer l'environnement virtuel Python
- Installer model2vec et toutes les dépendances
- Vérifier que tout fonctionne

### Étape 2: Configuration CPU Obligatoire

**⚠️ IMPORTANT:** Avec votre GPU 5GB, vous DEVEZ forcer le mode CPU

Éditez `distill_qwen25_7b.py` ligne 18:
```python
CONFIG = {
    "device": "cpu",  # ← Changez "cuda" en "cpu"
}
```

### Étape 3: Lancement de la Distillation (10-20h)

**Option A: Run overnight (recommandé)**
```bash
screen -S distill
./run_qwen25_7b_distillation.sh
# Détacher: Ctrl+A puis D
# Réattacher demain: screen -r distill
```

**Option B: Run en background**
```bash
nohup ./run_qwen25_7b_distillation.sh > distillation.log 2>&1 &
# Surveiller: tail -f distillation.log
```

### Étape 4: Validation (Le Lendemain)

**Tests (2 min):**
```bash
./test_qwen25_7b_model.sh
```

**Évaluation (5 min):**
```bash
./evaluate_qwen25_7b.sh
```

**Si score ≥ 91%, déploiement (10 min):**
```bash
./deploy_qwen25_7b.sh
```

---

## ⏱️ Timeline Complète (Votre Machine)

```
Jour 1 - Soir:
  19:00 - Setup environnement           (5 min)
  19:05 - Éditer config pour CPU        (2 min)
  19:10 - Lancer distillation           (1 min)
  19:11 - ⏰ LAISSER TOURNER TOUTE LA NUIT

Jour 2 - Matin:
  07:00 - ✅ Distillation terminée
  07:05 - Tests du modèle               (2 min)
  07:10 - Évaluation complète            (5 min)
  07:15 - Déploiement si OK             (10 min)
  07:25 - 🎉 TERMINÉ!
```

**Total:** ~12-15 heures (dont 10-12h distillation CPU overnight)

---

## 📊 Résultats Attendus

### Métriques de Qualité (Target)

| Métrique | Baseline | Qwen2.5-7B | Amélioration |
|----------|----------|------------|--------------|
| **Overall** | 68.2% | **91-95%** | **+23-27%** |
| Instruction Awareness | 95.3% | 96-98% | +1-3% |
| Semantic Similarity | 95.0% | 96-98% | +1-3% |
| Code Understanding | 86.4% | 92-96% | +6-10% |
| Domain Knowledge | 65-70% | 88-92% | +18-25% |
| Multilingual | 60-65% | 85-90% | +20-28% |

### Spécifications du Modèle

```
Taille:      ~65MB (vs 14GB full = 215x plus petit)
Dimensions:  1024D
Vocabulaire: 32K tokens (Qwen tokenizer)
Vitesse:     500-1000x plus rapide
Latence:     <1ms par requête
Mémoire:     <512MB runtime
```

---

## 🎯 Pourquoi Qwen2.5-7B?

### Performance SOTA 2025

```
MMLU:      83.5%  (connaissances générales)
GSM8K:     93.6%  (raisonnement mathématique)
HumanEval: 89.5%  (génération de code)
```

### Avantages Uniques

✅ **Meilleur modèle Qwen2.5** disponible (vs 1.5B actuel)
✅ **Multilingue** - 29+ langues supportées
✅ **Code-aware** - Entraîné sur corpus code massif
✅ **Instruction-tuned** - Excellent pour RAG/Q&A
✅ **Long contexte** - 128K tokens (vs 32K pour autres)

### Avec Model2Vec

⚡ **500-1000x plus rapide** que le modèle complet
📦 **215x plus petit** (65MB vs 14GB)
💰 **10-100x moins cher** en coûts compute
🔋 **Edge-deployable** (mobile, IoT, embedded)

---

## 📋 Checklist Complète

### Avant de Commencer

- [ ] Lire ce fichier en entier
- [ ] Comprendre que ça va prendre 10-20h en CPU
- [ ] Avoir ~15h libres sur la machine (overnight OK)
- [ ] Fermer applications gourmandes en RAM

### Pendant la Configuration

- [ ] Exécuter: `./setup_qwen25_7b_env.sh`
- [ ] Vérifier: `source venv/bin/activate` fonctionne
- [ ] Éditer: `distill_qwen25_7b.py` → device="cpu"
- [ ] Choisir: screen ou nohup pour run overnight

### Pendant la Distillation

- [ ] Lancer en screen/nohup (pas en foreground!)
- [ ] Vérifier les premiers logs (5-10 min)
- [ ] S'assurer que ça télécharge Qwen2.5-7B
- [ ] Laisser tourner overnight

### Après la Distillation

- [ ] Vérifier que `models/qwen25-7b-deposium-1024d/` existe
- [ ] Vérifier la taille: `du -sh models/qwen25-7b-deposium-1024d/`
- [ ] Lancer tests: `./test_qwen25_7b_model.sh`
- [ ] Lancer éval: `./evaluate_qwen25_7b.sh`
- [ ] Si ≥ 91%, déployer: `./deploy_qwen25_7b.sh`

---

## 🆘 Résolution de Problèmes

### Problème 1: Out Of Memory pendant distillation

**Solution:**
```bash
# Vérifier que device="cpu" dans CONFIG
grep "device" distill_qwen25_7b.py

# Si toujours GPU, forcer:
export CUDA_VISIBLE_DEVICES=""
./run_qwen25_7b_distillation.sh
```

### Problème 2: Distillation très lente

**Réponse:** C'est normal en mode CPU!
- 10-20 heures attendu
- Vérifier avec `htop` que ça utilise bien le CPU
- Pas de panique, laissez tourner

### Problème 3: model2vec non installé

**Solution:**
```bash
source venv/bin/activate
pip install model2vec>=0.6.0
```

### Problème 4: Score d'évaluation < 91%

**Options:**
1. Re-distiller avec meilleur paramètres:
   ```python
   CONFIG = {
       "pca_dims": 1536,  # Au lieu de 1024
       "corpus_size": 2_000_000,  # Au lieu de 1M
   }
   ```

2. Accepter un score < 91% si > 85%
   - Toujours meilleur que baseline (68.2%)
   - Acceptable pour production

3. Utiliser une machine plus puissante
   - Cloud GPU 16GB+
   - Score optimal avec meilleur hardware

---

## 💡 Alternative Cloud (Si Urgent)

Si vous ne pouvez pas attendre 10-20h, utilisez le cloud:

### Option 1: AWS EC2 g5.xlarge
```
GPU:  24GB NVIDIA A10G
Prix: ~$1.00/h
Temps: 2-3 heures
Coût:  ~$2-3 total
```

### Option 2: Paperspace
```
GPU:  16GB RTX A4000
Prix: ~$0.76/h
Temps: 3-4 heures
Coût:  ~$2.50 total
```

### Procédure Cloud

1. Créer instance avec GPU 16GB+
2. `git clone` ce repo
3. `./setup_qwen25_7b_env.sh`
4. NE PAS éditer device (laissez GPU)
5. `./run_qwen25_7b_distillation.sh`
6. Après 2-3h, télécharger le modèle
7. Détruire l'instance

---

## 📚 Documentation Complète

### Pour Démarrage Rapide
```bash
cat QWEN25_7B_QUICKSTART.md
```

### Pour Référence Complète
```bash
cat QWEN25_7B_DISTILLATION_GUIDE.md
```

### Pour Checklist Détaillée
```bash
cat PRE_DISTILLATION_CHECKLIST.md
```

### Pour Vue d'Ensemble Technique
```bash
cat QWEN25_7B_README.md
```

---

## 🎯 Critères de Succès

### ✅ Prêt pour Production Si:

- Overall quality ≥ 91%
- Instruction awareness ≥ 95%
- Code understanding ≥ 90%
- Taille modèle ≤ 70MB
- Tous les tests passent
- Container Docker démarre

### ⚠️ Re-distillation Nécessaire Si:

- Overall quality < 85%
- Tests échouent
- Taille modèle > 100MB
- Erreurs pendant distillation

---

## 🔄 Workflow Complet

```
1. PRÉPARATION (✅ Terminé)
   ├── Scripts créés
   ├── Documentation écrite
   └── Configuration définie

2. SETUP (⏳ Prochain, 5 min)
   ├── Environnement virtuel
   ├── Installation dépendances
   └── Configuration CPU

3. DISTILLATION (⏳ 10-20h overnight)
   ├── Téléchargement Qwen2.5-7B
   ├── Conversion Model2Vec
   └── Sauvegarde modèle

4. VALIDATION (⏳ Le lendemain matin)
   ├── Tests fonctionnels
   ├── Évaluation qualité
   └── Vérification score ≥ 91%

5. DÉPLOIEMENT (⏳ Si validé)
   ├── Update API
   ├── Build Docker
   ├── Test container
   └── Push production

6. DOCUMENTATION (⏳ Après déploiement)
   ├── Update README
   ├── Add benchmarks
   └── Create summary
```

---

## 📞 Support

### Documentation
- Ce fichier (FR): Vue d'ensemble et instructions
- QWEN25_7B_QUICKSTART.md: Guide rapide (EN)
- QWEN25_7B_DISTILLATION_GUIDE.md: Référence complète (EN)

### Ressources Externes
- Model2Vec: https://github.com/MinishLab/model2vec
- Qwen2.5: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- HuggingFace Docs: https://huggingface.co/docs

---

## ✅ Statut Actuel

**Préparation:** ✅ Complète
**Configuration:** ⏳ À faire (5 min)
**Distillation:** ⏳ À lancer (10-20h)

---

## 🚀 DÉMARRER MAINTENANT

```bash
# Étape 1: Setup (5 min)
./setup_qwen25_7b_env.sh

# Étape 2: Éditer config (2 min)
nano distill_qwen25_7b.py
# Changer device: "cpu"

# Étape 3: Lancer distillation (10-20h)
screen -S distill
./run_qwen25_7b_distillation.sh
# Ctrl+A puis D pour détacher

# Étape 4: Le lendemain (7 min)
screen -r distill  # Vérifier logs
./test_qwen25_7b_model.sh
./evaluate_qwen25_7b.sh

# Étape 5: Si OK (10 min)
./deploy_qwen25_7b.sh
```

---

**Dernière mise à jour:** 2025-10-14
**Priorité:** 🔥 ABSOLUE
**Statut:** ✅ Prêt à démarrer
**Prochaine étape:** `./setup_qwen25_7b_env.sh`
