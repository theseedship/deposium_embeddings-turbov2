# 🔧 FIX: Correction du Space HuggingFace

**Problème:** `ImportError: cannot import name 'distill_model' from 'model2vec'`

**Cause:** Mauvaise API utilisée. `distill_model` n'existe pas !

**Solution:** Utiliser `distill` depuis `model2vec.distill`

---

## ✅ Corrections à Appliquer (5 minutes)

### 1. Corriger `requirements.txt`

Aller sur votre Space → Files → Edit `requirements.txt`

**REMPLACER:**
```txt
model2vec>=0.6.0
```

**PAR:**
```txt
model2vec[distill]>=0.6.0
```

**requirements.txt complet:**
```txt
model2vec[distill]>=0.6.0
torch>=2.0.0
transformers>=4.50.0
gradio>=4.0.0
numpy>=1.24.0
sentencepiece>=0.1.99
protobuf>=3.20.0
```

### 2. Corriger `app.py`

Aller sur votre Space → Files → Edit `app.py`

**Changements nécessaires:**

#### A. Ligne 9 - REMPLACER l'import:
```python
# AVANT (incorrect):
from model2vec import distill_model

# APRÈS (correct):
from model2vec.distill import distill
```

#### B. Lignes 65-75 - REMPLACER l'appel de fonction:
```python
# AVANT (incorrect):
model = distill_model(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    pca_dims=1024,
    apply_pca=True,
    use_subword=True,
    apply_zipf=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    show_progress_bar=True,
)

# APRÈS (correct):
model = distill(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    pca_dims=1024,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
```

#### C. Ligne 90 - REMPLACER l'appel encode:
```python
# AVANT:
embeddings = model.encode(test_texts, show_progress_bar=False)

# APRÈS:
embeddings = model.encode(test_texts)
```

### 3. Commit les Changements

Après avoir modifié les 2 fichiers:
1. Cliquer "Commit changes to main"
2. Attendre rebuild (5-10 min)
3. Relancer App → Start Distillation

---

## 📄 Ou: Remplacer Complètement

**Option plus simple:** Remplacer tout le contenu d'un coup

### app.py Complet Corrigé

Copier le contenu de:
```bash
cat huggingface_space_app_FIXED.py
```

Aller sur Space → Files → Edit `app.py` → Remplacer TOUT → Commit

### requirements.txt Complet Corrigé

Copier le contenu de:
```bash
cat requirements_FIXED.txt
```

Aller sur Space → Files → Edit `requirements.txt` → Remplacer TOUT → Commit

---

## ⏱️ Timeline Correction

```
Maintenant - Éditer requirements.txt          (1 min)
         - Éditer app.py                      (2 min)
         - Commit                             (1 min)
         - Attendre rebuild                   (5-10 min)
         - Relancer distillation              (1 min)
         - ⏰ ATTENDRE 30-60 MIN

Dans 1h  - ✅ Télécharger ZIP
         - Tester localement
```

**Coût:** Le temps déjà écoulé est perdu, mais on repart avec les bons paramètres.

---

## 🔍 Différences API Model2Vec

### ❌ Ancienne API (incorrecte):
```python
from model2vec import distill_model

model = distill_model(
    model_name="...",
    pca_dims=1024,
    apply_pca=True,
    use_subword=True,
    apply_zipf=True,
    device="cuda",
    show_progress_bar=True,
)
```

### ✅ Nouvelle API (correcte):
```python
from model2vec.distill import distill

model = distill(
    model_name="...",
    pca_dims=1024,
    device="cuda",
)
```

**Changements:**
- Import: `model2vec.distill` au lieu de `model2vec`
- Fonction: `distill()` au lieu de `distill_model()`
- Paramètres: Simplifiés (model2vec utilise des defaults optimaux)
- Installation: `model2vec[distill]` au lieu de `model2vec`

---

## 🆘 Si Ça Échoue Encore

### Vérifier les Versions

Dans app.py, ajouter au début:
```python
import model2vec
print(f"model2vec version: {model2vec.__version__}")

from model2vec.distill import distill
print("✅ distill imported successfully")
```

### Vérifier l'Installation

Dans requirements.txt, forcer la version:
```txt
model2vec[distill]==0.6.0
```

### Contacter Model2Vec

Si vraiment ça ne marche pas:
- GitHub: https://github.com/MinishLab/model2vec/issues
- Vérifier la doc: https://github.com/MinishLab/model2vec#distillation

---

## ✅ Checklist Correction

- [ ] Éditer requirements.txt → `model2vec[distill]>=0.6.0`
- [ ] Éditer app.py → Import `from model2vec.distill import distill`
- [ ] Éditer app.py → Appel `model = distill(...)`
- [ ] Éditer app.py → Encode `model.encode(texts)` sans show_progress_bar
- [ ] Commit changes
- [ ] Attendre rebuild
- [ ] Relancer App
- [ ] Vérifier logs pour "✅"
- [ ] Attendre 30-60 min
- [ ] Télécharger ZIP

---

## 💡 Pour Nemotron Aussi

**IMPORTANT:** Appliquer les mêmes corrections pour Nemotron !

Quand vous créerez le Space Nemotron:
- ✅ Utiliser `huggingface_nemotron_app.py` avec API corrigée
- ✅ Utiliser `requirements_FIXED.txt`
- ✅ Ou corriger manuellement avec les mêmes changements

---

**Status:** 🔧 Correction en cours
**ETA:** 5-10 min rebuild + 30-60 min distillation
**Coût perdu:** ~$0.20-0.30 (15-20 min au tarif A10G small)
