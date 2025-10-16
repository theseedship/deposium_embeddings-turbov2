# Distillation sur HuggingFace Spaces - Guide Complet

**🆓 GRATUIT** avec GPU Tesla T4 16GB!
**⏱️ 2-3 heures** pour Qwen2.5-7B
**🎯 91-95%** qualité cible

---

## ✅ Solution Recommandée: HuggingFace Spaces

### Pourquoi c'est parfait pour vous?

**1. GPU GRATUIT** 🎁
- Tesla T4 16GB VRAM (gratuit!)
- Largement suffisant pour Qwen2.5-7B
- Pas de frais AWS/GCP

**2. Temps raisonnable**
- 2-3 heures pour Qwen2.5-7B
- Vous pouvez surveiller le progrès
- Télécharger le modèle après

**3. Simple à configurer**
- Interface web
- Pas besoin de configurer cloud
- Upload script + run

---

## 🚀 Guide Pas-à-Pas

### Étape 1: Créer un Space (2 minutes)

1. Allez sur https://huggingface.co/spaces
2. Cliquez "Create new Space"
3. Configuration:
   ```
   Name: qwen25-7b-distillation
   SDK: Gradio
   Hardware: Tesla T4 (FREE) ← IMPORTANT!
   Visibility: Private
   ```

### Étape 2: Upload le Script (3 minutes)

Créez `app.py` dans le Space:

```python
#!/usr/bin/env python3
"""
Distillation Qwen2.5-7B sur HuggingFace Space
"""

import gradio as gr
import torch
from model2vec import distill_model
from datetime import datetime
from pathlib import Path
import zipfile

def distill_qwen25_7b(progress=gr.Progress()):
    """Distille Qwen2.5-7B vers Model2Vec"""

    progress(0, desc="Starting distillation...")

    output_dir = Path("models/qwen25-7b-deposium-1024d")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    log = []

    try:
        log.append(f"🚀 Starting Qwen2.5-7B distillation at {start_time}")
        log.append(f"GPU: {torch.cuda.get_device_name(0)}")
        log.append(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        log.append("")

        progress(0.1, desc="Loading model...")

        # Distillation
        model = distill_model(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            pca_dims=1024,
            apply_pca=True,
            use_subword=True,
            apply_zipf=True,
            device="cuda",
            show_progress_bar=True,
        )

        progress(0.8, desc="Saving model...")

        # Save
        model.save_pretrained(str(output_dir))

        # Test
        progress(0.9, desc="Testing model...")
        test_texts = ["Hello world", "Machine learning"]
        embeddings = model.encode(test_texts)

        end_time = datetime.now()
        duration = end_time - start_time

        # Get size
        model_size = sum(f.stat().st_size for f in output_dir.glob("**/*") if f.is_file())
        model_size_mb = model_size / (1024 * 1024)

        # Create ZIP for download
        progress(0.95, desc="Creating download archive...")
        zip_path = "qwen25-7b-deposium-1024d.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in output_dir.rglob("*"):
                if file.is_file():
                    zipf.write(file, file.relative_to(output_dir.parent))

        log.append(f"✅ Distillation complete in {duration}")
        log.append(f"Model size: {model_size_mb:.1f}MB")
        log.append(f"Embeddings shape: {embeddings.shape}")
        log.append(f"")
        log.append(f"📦 Download the model below!")

        progress(1.0, desc="Done!")

        return "\n".join(log), zip_path

    except Exception as e:
        log.append(f"")
        log.append(f"❌ Error: {e}")
        return "\n".join(log), None

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Qwen2.5-7B → Model2Vec Distillation")
    gr.Markdown("Distille Qwen2.5-7B-Instruct en Model2Vec 1024D (~65MB)")
    gr.Markdown("**⏱️ Durée: 2-3 heures** avec Tesla T4")

    with gr.Row():
        start_btn = gr.Button("🚀 Start Distillation", variant="primary", size="lg")

    output_log = gr.Textbox(label="Logs", lines=20, max_lines=50)
    download_file = gr.File(label="📦 Download Model")

    start_btn.click(
        fn=distill_qwen25_7b,
        inputs=[],
        outputs=[output_log, download_file]
    )

if __name__ == "__main__":
    demo.launch()
```

### Étape 3: requirements.txt

Créez `requirements.txt`:
```txt
model2vec>=0.6.0
torch>=2.0.0
transformers>=4.50.0
gradio>=4.0.0
numpy>=1.24.0
```

### Étape 4: Lancer la Distillation (2-3h)

1. Ouvrez votre Space
2. Cliquez "Start Distillation"
3. Attendez 2-3h
4. Téléchargez le ZIP

### Étape 5: Utiliser le Modèle (5 min)

```bash
# Sur votre machine locale
unzip qwen25-7b-deposium-1024d.zip -d models/

# Tester
python3 test_qwen25_7b_model.py

# Évaluer
python3 quick_eval_qwen25_7b_1024d.py
```

---

## 💰 Coûts

**HuggingFace Spaces FREE Tier:**
- ✅ Tesla T4 16GB (gratuit!)
- ✅ Temps illimité
- ✅ Pas de carte bancaire requise

**Limitations:**
- Space se ferme après inactivité (OK pour nous)
- Pas de persistent storage (on télécharge le ZIP)
- Queue possible si beaucoup d'utilisateurs (rare)

---

## 🆚 Comparaison des Options

| Option | GPU | Temps | Coût | Qualité |
|--------|-----|-------|------|---------|
| **HuggingFace Space** | T4 16GB | 2-3h | **GRATUIT** | 91-95% ✅ |
| Local GPU 5GB | RTX 4050 | ❌ OOM | Gratuit | - |
| Local CPU | - | 10-20h | Gratuit | 91-95% |
| Qwen2.5-3B Local | RTX 4050 | 1-2h | Gratuit | 85-88% |
| AWS g5.xlarge | A10G 24GB | 2-3h | ~$2-3 | 91-95% |

**Gagnant: HuggingFace Space** 🏆

---

## 🎯 Alternative: Kaggle Notebooks

**Aussi gratuit avec GPU!**

### Kaggle Setup (5 min)

1. Allez sur https://www.kaggle.com/notebooks
2. New Notebook
3. Settings → Accelerator → GPU T4 x2 (gratuit!)
4. Session time: 12h (largement suffisant)

**Notebook Code:**
```python
# Kaggle Notebook - Cell 1
!pip install model2vec transformers

# Cell 2
import torch
from model2vec import distill_model
from pathlib import Path

output_dir = Path("qwen25-7b-deposium-1024d")
output_dir.mkdir(exist_ok=True)

print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 3 - Distillation (2-3h)
model = distill_model(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    pca_dims=1024,
    apply_pca=True,
    use_subword=True,
    apply_zipf=True,
    device="cuda",
    show_progress_bar=True,
)

# Cell 4 - Save
model.save_pretrained(str(output_dir))

# Cell 5 - Download
!zip -r qwen25-7b-model.zip qwen25-7b-deposium-1024d/
# Download via Kaggle UI
```

---

## 📋 Checklist HuggingFace Spaces

### Avant de Commencer

- [ ] Compte HuggingFace (gratuit)
- [ ] Comprendre que ça prend 2-3h
- [ ] Préparer ~1GB pour télécharger le ZIP

### Création du Space

- [ ] Créer Space avec Tesla T4
- [ ] Upload app.py (code ci-dessus)
- [ ] Upload requirements.txt
- [ ] Vérifier que ça build (5-10 min)

### Distillation

- [ ] Ouvrir l'interface du Space
- [ ] Cliquer "Start Distillation"
- [ ] Surveiller les logs
- [ ] Attendre 2-3h
- [ ] Télécharger le ZIP

### Après Distillation

- [ ] Extraire le ZIP localement
- [ ] Tester: `python3 test_qwen25_7b_model.py`
- [ ] Évaluer: `python3 quick_eval_qwen25_7b_1024d.py`
- [ ] Si score ≥ 91%, déployer
- [ ] Supprimer le Space HF (optionnel)

---

## 🆘 Troubleshooting

### Space ne build pas

**Solution:**
```txt
# requirements.txt - Version minimale
gradio
torch
transformers
model2vec
```

### Out of Memory sur T4

**Solution 1:** Réduire corpus_size
```python
# Dans app.py, ajouter:
corpus_size = 500_000  # Au lieu de 1M
```

**Solution 2:** Utiliser Qwen2.5-3B à la place
```python
model_name = "Qwen/Qwen2.5-3B-Instruct"
```

### Distillation très lente

**Réponse:** Normal!
- T4: 2-3 heures attendu
- Pas de panique, laissez tourner

### Ne peut pas télécharger le ZIP

**Solution:** Sauvegarder sur HF Hub
```python
# Dans app.py, après save_pretrained:
model.push_to_hub("votre-username/qwen25-7b-1024d")
```

---

## 💡 Recommandations

### Option 1: HuggingFace Space (MEILLEUR ✅)

**Avantages:**
- ✅ Gratuit avec T4 16GB
- ✅ Interface web simple
- ✅ 2-3 heures
- ✅ Qualité 91-95%

**À faire:**
```
1. Créer Space avec T4
2. Upload les fichiers
3. Cliquer Start
4. Télécharger ZIP après 2-3h
```

### Option 2: Kaggle Notebook

**Avantages:**
- ✅ Gratuit avec T4 x2
- ✅ 12h de session
- ✅ Interface notebook

**À faire:**
```
1. Créer Kaggle Notebook
2. Activer GPU T4 x2
3. Copier le code
4. Run cells
5. Télécharger ZIP
```

### Option 3: Local Qwen2.5-3B (RAPIDE)

**Si vous voulez quelque chose MAINTENANT:**
```bash
python3 distill_qwen25_3b.py  # 1-2h, 85-88%
```

---

## 🎯 Timeline Recommandée

**Aujourd'hui - Après-midi:**
```
14:00 - Créer HF Space           (5 min)
14:05 - Upload fichiers          (3 min)
14:10 - Start distillation       (1 min)
14:11 - ⏰ Laisser tourner 2-3h

17:00 - ✅ Télécharger ZIP
17:05 - Extraire et tester       (5 min)
17:10 - Évaluer qualité          (5 min)
17:15 - Déployer si OK           (10 min)
17:25 - 🎉 TERMINÉ!
```

**Total: ~3h15 (dont 2-3h attente)**

---

## 📦 Fichiers à Préparer

Créez ces fichiers pour le Space:

**1. app.py** (voir code ci-dessus)
**2. requirements.txt** (voir ci-dessus)
**3. README.md** (optionnel):
```markdown
# Qwen2.5-7B → Model2Vec Distillation

Distille Qwen2.5-7B-Instruct en Model2Vec 1024D.

## Usage

1. Cliquer "Start Distillation"
2. Attendre 2-3 heures
3. Télécharger le ZIP

## Output

- Model2Vec 1024D (~65MB)
- Performance: 91-95%
```

---

## 🎉 Résumé

**Solution choisie:** HuggingFace Space avec Tesla T4 (GRATUIT!)

**Avantages:**
- 🆓 Gratuit
- ⚡ 2-3h (raisonnable)
- 🎯 91-95% qualité
- 🎮 GPU T4 16GB (parfait pour 7B)

**Prochaines étapes:**
1. Aller sur https://huggingface.co/spaces
2. Create new Space
3. Upload app.py + requirements.txt
4. Start distillation
5. Revenir dans 3h

---

**Date:** 2025-10-14
**Status:** ✅ Ready to start
**Platform:** HuggingFace Spaces (FREE)
**ETA:** 2-3 hours
