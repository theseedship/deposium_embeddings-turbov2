# Railway Private Network - Analyse du Problème

## ❌ Problème Rencontré

**URL testée:** `http://deposium_embeddings-turbov2.railway.internal:11435`
**Erreur:** `ECONNREFUSED fd12:d339:b767:0:1000:41:d1cb:8f10:11435`

L'URL interne Railway ne fonctionne pas entre N8N et TurboX.v2.

---

## 🔍 Analyse Technique

### Railway Private Networking - Comment ça fonctionne

Railway propose deux modes de communication:

1. **Public URL (HTTPS):**
   - Format: `https://servicename.up.railway.app`
   - Accessible depuis internet
   - Certificat SSL automatique
   - ✅ **Fonctionne toujours**

2. **Private Network (Interne):**
   - Format: `http://servicename.railway.internal`
   - Accessible uniquement entre services Railway
   - Réseau IPv6 interne
   - ❌ **Conditions strictes**

### Pourquoi ça échoue ?

#### Raison #1: Services dans des projets différents

**Railway Private Network fonctionne UNIQUEMENT au sein d'un même projet.**

Si N8N et TurboX.v2 sont dans des projets Railway séparés:
- Le DNS résout (`fd12:...` prouve que la résolution fonctionne)
- Mais la connexion est **refusée** (firewall inter-projets)

**Solution:** Déployer les deux services dans le **même projet Railway**

#### Raison #2: Port non routé par Private Network

Railway Private Network route via un **proxy HTTP interne**.

Format attendu:
- ✅ `http://servicename.railway.internal` (sans port)
- ❌ `http://servicename.railway.internal:11435` (avec port)

Railway détecte automatiquement le port exposé (11435) et route le trafic.

**Solution:** Tester **sans port** dans l'URL

#### Raison #3: Private Networking pas activé

Railway Pro/Teams peut nécessiter l'activation explicite de Private Networking.

**Vérification:**
1. Aller dans Railway Dashboard → Project Settings
2. Chercher "Private Networking" ou "Internal Networking"
3. Activer si option disponible

---

## ✅ Solutions Proposées

### Solution 1: Même Projet Railway (Recommandée)

**Si services dans projets séparés:**

1. Créer nouveau service TurboX.v2 dans le projet N8N
2. Supprimer l'ancien service du projet séparé
3. Utiliser: `http://deposium-embeddings-turbov2.railway.internal`

**Avantages:**
- ✅ Private Network fonctionne
- ✅ Pas d'exposition publique
- ✅ Latence réduite
- ✅ Gratuit (pas de coût bande passante)

**Inconvénients:**
- ⚠️ Tous les services dans un seul projet (moins modulaire)

### Solution 2: URL Publique avec Authentification

**Garder l'URL publique HTTPS mais ajouter sécurité:**

```python
# Ajouter dans src/main.py
import os
from fastapi import Header, HTTPException

API_KEY = os.getenv("TURBOV2_API_KEY", "")

async def verify_api_key(x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_api_key

@app.post("/api/embed", dependencies=[Depends(verify_api_key)])
async def create_embedding(request: EmbedRequest):
    # ... existing code
```

**Configuration Railway:**
```bash
TURBOV2_API_KEY=your-secure-random-key-here
```

**Configuration N8N:**
```yaml
Base URL: https://deposiumembeddings-turbov2-staging.up.railway.app
Headers:
  X-API-Key: your-secure-random-key-here
```

**Avantages:**
- ✅ Fonctionne immédiatement
- ✅ Sécurisé avec API Key
- ✅ Projets séparés OK
- ✅ Gratuit pour trafic interne Railway

**Inconvénients:**
- ⚠️ Exposé publiquement (mais protégé)
- ⚠️ Très légèrement plus de latence

### Solution 3: Test URL Interne Sans Port

**Si services dans le même projet:**

Tester dans N8N credentials:
```
Base URL: http://deposium-embeddings-turbov2.railway.internal
Model: turbov2
```

**Sans spécifier le port 11435** - Railway route automatiquement.

**Test de diagnostic:**
```bash
# Dans un workflow N8N, node HTTP Request:
GET http://deposium-embeddings-turbov2.railway.internal/health
```

Si ça fonctionne → Utiliser cette URL
Si ça échoue → Services dans projets différents

### Solution 4: Railway Teams + Shared Private Network

**Pour organisations avec Railway Teams:**

Railway Teams permet le Private Networking entre projets.

**Coût:** ~$20/mois (plan Teams minimum)

**Setup:**
1. Upgrade vers Railway Teams
2. Créer Shared Private Network
3. Ajouter les deux projets au réseau
4. Utiliser `.railway.internal` URLs

---

## 🎯 Recommandation Finale

### Pour Dev/Test
**Utiliser URL publique HTTPS** (solution actuelle)
- Simple, fonctionne toujours
- Ajouter API Key si besoin de sécurité
- Coût: $0

### Pour Production
**Option A:** Migrer TurboX.v2 dans projet N8N
- Private Network garanti
- Pas d'exposition publique
- Latence minimale

**Option B:** API Key + URL publique
- Plus modulaire
- Sécurisé suffisamment
- Plus flexible

---

## 🧪 Diagnostic Checklist

Pour comprendre pourquoi Private Network échoue:

```bash
# 1. Vérifier projets Railway
railway status
# → Noter: Project ID pour N8N et TurboX.v2

# 2. Si même project ID:
#    → Tester sans port: http://servicename.railway.internal
# 3. Si projets différents:
#    → Private Network NE PEUT PAS fonctionner
#    → Utiliser URL publique ou migrer service

# 4. Vérifier résolution DNS (depuis N8N workflow):
nslookup deposium-embeddings-turbov2.railway.internal
# → Si résout vers IPv6: DNS OK, mais firewall bloque
# → Si ne résout pas: Service dans autre projet

# 5. Test ultime (HTTP Request node N8N):
GET http://deposium-embeddings-turbov2.railway.internal/health
# → Success: Private Network fonctionne
# → ECONNREFUSED: Projets séparés ou port blocking
```

---

## 📊 Comparaison des Solutions

| Solution | Coût | Sécurité | Latence | Complexité |
|----------|------|----------|---------|------------|
| **Même projet** | $0 | ★★★★★ | ★★★★★ | ★★★☆☆ |
| **URL publique** | $0 | ★★★☆☆ | ★★★★☆ | ★☆☆☆☆ |
| **URL + API Key** | $0 | ★★★★☆ | ★★★★☆ | ★★☆☆☆ |
| **Teams Network** | $20/mo | ★★★★★ | ★★★★★ | ★★★★☆ |

---

## 🔐 Sécurité - URL Publique

**"L'URL publique me dérange" - Pourquoi c'est OK:**

1. **HTTPS** → Trafic chiffré
2. **Aucun lien public** → Pas de découverte via moteurs de recherche
3. **API Key** → Accès restreint
4. **Railway internal routing** → Trafic entre services Railway reste interne (même via URL publique)
5. **Rate limiting** → Peut être ajouté

**En production:**
- Ajouter API Key (5 min de setup)
- Monitorer access logs Railway
- Optionnel: IP whitelist si N8N a IP fixe

---

## 📝 Prochaines Étapes

### Test Immédiat (5 min)
1. Vérifier si N8N et TurboX.v2 dans même projet Railway
2. Si oui: Tester `http://deposium-embeddings-turbov2.railway.internal` (sans port)
3. Si ça marche: Utiliser cette URL
4. Si ça échoue: Passer à solution B

### Solution Temporaire (actuelle)
- ✅ Garder `https://deposiumembeddings-turbov2-staging.up.railway.app`
- ✅ Fonctionne parfaitement
- ⏳ Ajouter API Key si besoin sécurité

### Solution Permanente (choix à faire)
- **Option 1:** Migrer service dans projet N8N (Private Network)
- **Option 2:** Ajouter API Key + garder URL publique

---

**Dernière mise à jour:** 2025-10-09
**Status actuel:** URL publique HTTPS fonctionne ✅
**Besoin:** Private Network (optionnel, dépend de l'architecture Railway)
