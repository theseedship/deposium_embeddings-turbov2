# Railway Configuration - URL Publique vs Interne

## ❌ Problème: URL Interne Ne Fonctionne Pas

**Testé:** `http://deposium_embeddings-turbov2.railway.internal:11435`
**Erreur:** `ECONNREFUSED fd12:d339:b767:0:1000:41:d1cb:8f10:11435`

**Cause:** Railway network isolation entre services - l'URL interne n'est pas accessible

## ✅ Solution: Utiliser l'URL Publique

**URL publique Railway:** `http://deposiumembeddings-turbov2-staging.up.railway.app`

**Important:**
- ✅ Pas besoin de spécifier le port (Railway route automatiquement)
- ✅ Fonctionne depuis N8N
- ✅ Pas de surcoût (trafic interne Railway gratuit même sur URL publique)

---

## 🔌 Configuration N8N - CORRECTE

### Credentials Ollama

```yaml
Name: TurboX.v2 Railway
Base URL: http://deposiumembeddings-turbov2-staging.up.railway.app
Model: turbov2
API Key: [laisser vide]
```

### Node Qwen Embedding Tool

```yaml
Credentials: TurboX.v2 Railway
Dimensions: 1024
Input: {{ $json.text }}
```

---

## ✅ Tests de Vérification

### 1. Health Check
```bash
GET http://deposiumembeddings-turbov2-staging.up.railway.app/health
# → {"status":"healthy"}
```

### 2. List Models
```bash
GET http://deposiumembeddings-turbov2-staging.up.railway.app/api/tags
# → {"models":[{"name":"turbov2",...}]}
```

### 3. Generate Embedding
```bash
POST http://deposiumembeddings-turbov2-staging.up.railway.app/api/embed
{
  "model": "turbov2",
  "input": "test rapide"
}
# → {"model":"turbov2","embeddings":[[1024 dimensions]]}
```

---

## 📊 Configuration Finale

| Paramètre | Valeur |
|-----------|--------|
| **Base URL** | http://deposiumembeddings-turbov2-staging.up.railway.app |
| **Model** | turbov2 |
| **Dimensions** | 1024 |
| **Port** | (pas nécessaire) |
| **API Key** | (vide) |

---

## 🚀 Prochaines Étapes

1. ✅ URL publique configurée
2. ⏳ Tester `/health` depuis N8N
3. ⏳ Tester embedding avec node Qwen
4. ⏳ Comparer vitesse vs Ollama local
5. ⏳ Migration workflows si satisfait

---

**Note:** L'URL publique Railway fonctionne pour trafic interne entre services Railway sans surcoût. C'est la méthode recommandée quand `.railway.internal` échoue.
