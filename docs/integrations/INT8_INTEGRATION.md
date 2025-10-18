# 🔄 Intégration du Modèle int8 (256D)

## 📊 Vue d'Ensemble

**Ajout du second modèle C10X/int8 au service d'embeddings existant.**

### Modèles Disponibles

| Modèle | Dimensions | Base | Utilisation |
|--------|------------|------|-------------|
| **turbov2** | 1024D | Qwen3-Embedding-0.6B | Embeddings généraux, recherche sémantique |
| **int8** | 256D | Qwen3-Reranker-0.6B | Embeddings légers, optimisation reranking |

---

## 🛠️ Modifications Apportées

### 1. `src/main.py`

**Changements principaux:**

```python
# Avant: Un seul modèle
model = StaticModel.from_pretrained("C10X/Qwen3-Embedding-TurboX.v2")

# Après: Deux modèles dans un dictionnaire
models = {
    "turbov2": StaticModel.from_pretrained("C10X/Qwen3-Embedding-TurboX.v2"),
    "int8": StaticModel.from_pretrained("C10X/int8")
}
```

**Endpoint `/api/embed` modifié:**
- Validation du modèle sélectionné
- Sélection dynamique du modèle
- Logging des dimensions générées

**Endpoints mis à jour:**
- `GET /` - Affiche les 2 modèles disponibles
- `GET /health` - Vérifie le chargement des 2 modèles
- `GET /api/tags` - Liste les 2 modèles (Ollama-compatible)

### 2. `README.md`

**Documentation complète:**
- Section "Available Models" avec détails des 2 modèles
- Exemples de curl pour chaque modèle
- Configuration N8N pour les 2 modèles
- Dimensions clarifiées (1024D vs 256D)

---

## 🧪 Tests de Validation

### Test Local

```bash
# 1. Rebuild du container
docker build -t deposium-embeddings-turbov2 .

# 2. Démarrer le service
docker run -p 11435:11435 deposium-embeddings-turbov2

# 3. Vérifier le chargement des modèles
curl http://localhost:11435/health

# Expected:
{
  "status": "healthy",
  "models_loaded": ["turbov2", "int8"]
}

# 4. Lister les modèles
curl http://localhost:11435/api/tags

# Expected: 2 modèles listés

# 5. Test TurboX.v2 (1024D)
curl -X POST http://localhost:11435/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"turbov2","input":"test"}'

# Expected: embeddings avec 1024 dimensions

# 6. Test int8 (256D)
curl -X POST http://localhost:11435/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"int8","input":"test"}'

# Expected: embeddings avec 256 dimensions
```

### Test Railway

```bash
# 1. Push to GitHub
git add .
git commit -m "feat: add int8 (256D) model support"
git push origin main

# 2. Railway auto-deploy

# 3. Test HTTPS
curl -X POST https://deposiumembeddings-turbov2-staging.up.railway.app/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"int8","input":"test"}'

# 4. Test Private Network (sans port)
curl -X POST http://deposium-embeddings-turbov2.railway.internal/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"int8","input":"test"}'
```

---

## 📝 Configuration N8N

### Créer 2 Credentials Ollama

**Credential 1: TurboX.v2 (1024D)**
```
Name: Deposium Embeddings - TurboX.v2
Base URL: http://deposium-embeddings-turbov2:11435
Model: turbov2
```

**Credential 2: int8 (256D)**
```
Name: Deposium Embeddings - int8
Base URL: http://deposium-embeddings-turbov2:11435
Model: int8
```

### Cas d'Usage

**Utiliser TurboX.v2 (1024D) pour:**
- Recherche sémantique principale
- Embeddings de haute qualité
- Tâches nécessitant plus de dimensions

**Utiliser int8 (256D) pour:**
- Embeddings légers (économie de stockage)
- Pré-filtrage rapide avant reranking
- Cas où 256D suffisent (classification simple)

---

## 🔍 Architecture Technique

### Pipeline int8

```
Input Text
    ↓
Qwen3-Reranker-0.6B Tokenizer
    ↓
Model2Vec Static Model
    ↓ PCA: 2048D → 256D
    ↓ SIF Weighting
    ↓ L2 Normalization
256D Embedding Vector
```

### Différences avec TurboX.v2

| Aspect | TurboX.v2 | int8 |
|--------|-----------|------|
| Tokenizer | Qwen3-Embedding-0.6B | Qwen3-Reranker-0.6B |
| PCA Source | 2048D | 2048D |
| Output Dims | 1024D | 256D |
| Use Case | Général | Reranking/Léger |

---

## 🚀 Prochaines Étapes

1. ✅ Code mis à jour avec dual model support
2. ✅ Documentation complète
3. ⏳ **Test local** - Vérifier les 2 modèles
4. ⏳ **Deployment Railway** - Push et test
5. ⏳ **Intégration N8N** - Créer les 2 credentials
6. ⏳ **Test workflow** - Vérifier les 2 modèles dans N8N
7. ⏳ **Plugin N8N rerank** - Créer le node custom (plus tard)

---

## 📚 Références

- **HuggingFace int8:** https://huggingface.co/C10X/int8
- **HuggingFace TurboX.v2:** https://huggingface.co/C10X/Qwen3-Embedding-TurboX.v2
- **Model2Vec:** https://github.com/MinishLab/model2vec
- **Config int8:** Tokenizer = Qwen/Qwen3-Reranker-0.6B, 256D output

---

*Intégration créée: 2025-10-09*
*Service: Dual Model Embeddings (1024D + 256D)*
*Stack: FastAPI + Model2Vec + Docker*
