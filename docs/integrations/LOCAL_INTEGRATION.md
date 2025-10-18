# Intégration Locale - TurboX.v2 dans deposium-local

## ✅ Intégration Complète

TurboX.v2 est maintenant intégré dans la stack locale deposium-local !

### 📦 Modifications Effectuées

#### 1. Docker Compose (`docker-compose.yml`)
**Service ajouté après Ollama (ligne 1052):**
```yaml
embeddings-turbov2:
  image: deposium-embeddings-turbov2:latest
  container_name: deposium-embeddings-turbov2
  networks:
    - deposium-internal
    - traefik
  # Port 11435 accessible internally via http://deposium-embeddings-turbov2:11435
  # External access via Traefik: http://turbov2.localhost
  labels:
    - "traefik.enable=true"
    - "traefik.http.routers.turbov2.rule=Host(`turbov2.localhost`)"
    - "traefik.http.services.turbov2.loadbalancer.server.port=11435"
    - "traefik.docker.network=traefik"
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:11435/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 10s
```

#### 2. VS Code Workspace (`deposium.code-workspace`)
**Nouveau dossier ajouté:**
```json
{
  "name": "🚀 TurboX.v2 Embeddings (Ultra-fast CPU)",
  "path": "../deposium_embeddings-turbov2"
}
```

---

## 🚀 Accès au Service

### URLs Disponibles

| Type | URL | Usage |
|------|-----|-------|
| **Interne N8N** | `http://deposium-embeddings-turbov2:11435` | Depuis workflows N8N |
| **Traefik** | `http://turbov2.localhost` | Accès externe via navigateur |
| **Health Check** | `http://deposium-embeddings-turbov2:11435/health` | Monitoring |
| **API Embed** | `http://deposium-embeddings-turbov2:11435/api/embed` | Génération embeddings |

### Test de Connexion

```bash
# Depuis l'hôte (via Traefik)
curl http://turbov2.localhost/health
# → {"status":"healthy"}

# Depuis un container (réseau deposium-internal)
docker run --rm --network deposium-local_deposium-internal curlimages/curl \
  http://deposium-embeddings-turbov2:11435/health
# → {"status":"healthy"}
```

---

## 🔌 Configuration N8N Local

### Credentials Ollama

**Créer nouveau credential dans N8N:**

```yaml
Name: TurboX.v2 Local
Type: Ollama API
Base URL: http://deposium-embeddings-turbov2:11435
Model: turbov2
API Key: [laisser vide]
```

### Node Qwen Embedding Tool

```yaml
Credentials: TurboX.v2 Local
Dimensions: 1024
Input: {{ $json.text }}
```

### Workflow de Test

1. Créer workflow N8N
2. Ajouter node "HTTP Request":
   ```
   Method: POST
   URL: http://deposium-embeddings-turbov2:11435/api/embed
   Body: {"model":"turbov2","input":"test local"}
   ```
3. Exécuter → Devrait retourner 1024 dimensions en <20ms

---

## 🛠️ Gestion du Service

### Commandes Make

```bash
# Démarrer toute la stack (inclut TurboX.v2)
make up

# Démarrer uniquement TurboX.v2
docker-compose up -d embeddings-turbov2

# Voir les logs
docker-compose logs -f embeddings-turbov2

# Restart le service
docker-compose restart embeddings-turbov2

# Arrêter le service
docker-compose stop embeddings-turbov2

# Supprimer le container
docker-compose rm -f embeddings-turbov2
```

### Commandes Docker Compose Directes

```bash
# Status
docker-compose ps embeddings-turbov2

# Logs
docker logs deposium-embeddings-turbov2 --tail 50

# Shell dans le container
docker exec -it deposium-embeddings-turbov2 sh

# Health check
docker inspect deposium-embeddings-turbov2 --format='{{.State.Health.Status}}'
```

---

## 🔍 Troubleshooting

### Service ne démarre pas

```bash
# Vérifier les logs
docker-compose logs embeddings-turbov2

# Vérifier l'image existe
docker images | grep turbov2

# Rebuild si nécessaire
cd /home/nico/code_source/tss/deposium_embeddings-turbov2
docker build -t deposium-embeddings-turbov2:latest .
```

### N8N ne peut pas accéder au service

```bash
# Vérifier réseau
docker inspect deposium-embeddings-turbov2 | grep -A 10 Networks

# Tester depuis N8N container
docker exec deposium-n8n-primary wget -qO- http://deposium-embeddings-turbov2:11435/health

# Vérifier que N8N et TurboX.v2 sont sur deposium-internal
docker network inspect deposium-local_deposium-internal
```

### Port 11435 déjà utilisé

```bash
# Trouver le processus
sudo lsof -i :11435

# Ou arrêter l'ancien container de test
docker stop deposium-embeddings-turbov2-test
docker rm deposium-embeddings-turbov2-test
```

---

## 📊 Performance Locale vs Railway

| Métrique | Local | Railway |
|----------|-------|---------|
| **Latency** | 4-16ms | 5-50ms |
| **Network** | Internal Docker | Internet HTTPS |
| **Startup** | <10s | <30s |
| **Availability** | Dépend de la stack | 99.9% uptime |

**Recommandation:**
- **Dev/Test:** Utiliser local (plus rapide)
- **Production:** Utiliser Railway (toujours disponible)

---

## 🔄 Mise à Jour du Service

### Rebuild Image

```bash
# 1. Aller dans le repo
cd /home/nico/code_source/tss/deposium_embeddings-turbov2

# 2. Pull dernières changes
git pull origin main

# 3. Rebuild image
docker build -t deposium-embeddings-turbov2:latest .

# 4. Restart le service
docker-compose restart embeddings-turbov2
```

### Changement de Modèle

Si vous voulez tester un autre modèle Model2Vec:

1. Modifier `src/main.py`:
   ```python
   model = StaticModel.from_pretrained("autre-modele-ici")
   ```

2. Rebuild et redémarrer:
   ```bash
   docker build -t deposium-embeddings-turbov2:latest .
   docker-compose restart embeddings-turbov2
   ```

---

## 🌐 Accès Traefik (turbov2.localhost)

### Configuration DNS Local

**Ajouter à `/etc/hosts` (ou `C:\Windows\System32\drivers\etc\hosts`):**
```
127.0.0.1 turbov2.localhost
```

### Test Navigateur

Ouvrir: http://turbov2.localhost

**Devrait afficher:**
```json
{
  "service": "Deposium Embeddings TurboX.v2",
  "status": "running",
  "model": "C10X/Qwen3-Embedding-TurboX.v2"
}
```

### Test API via Traefik

```bash
curl -X POST http://turbov2.localhost/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"turbov2","input":"test via traefik"}'
```

---

## 🔐 Sécurité Locale

**Pas de sécurité nécessaire en local:**
- Réseau Docker isolé
- Pas d'exposition internet
- Accès uniquement depuis conteneurs locaux

**Pour production, voir:**
- [RAILWAY_PRIVATE_NETWORK.md](RAILWAY_PRIVATE_NETWORK.md) - Options de sécurité Railway
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - API Key authentication

---

## 📝 Checklist d'Intégration

- [x] ✅ Service ajouté à `docker-compose.yml`
- [x] ✅ Workspace VS Code mis à jour
- [x] ✅ Service démarré et healthy
- [x] ✅ Réseau deposium-internal configuré
- [x] ✅ Traefik routing configuré
- [x] ✅ Tests de connexion validés
- [ ] ⏳ Credentials N8N créés
- [ ] ⏳ Workflow N8N testé

---

## 🎯 Prochaines Étapes

1. **Créer credentials N8N local** (2 min)
   - Base URL: `http://deposium-embeddings-turbov2:11435`
   - Model: `turbov2`

2. **Tester workflow N8N** (5 min)
   - Comparer vitesse vs Ollama
   - Vérifier qualité embeddings

3. **Migration progressive** (optionnel)
   - Remplacer Ollama par TurboX.v2 dans workflows existants
   - Monitorer performance

---

**Status:** ✅ Intégration locale complète et fonctionnelle
**Service URL:** `http://deposium-embeddings-turbov2:11435`
**Traefik URL:** `http://turbov2.localhost`
**Prêt pour:** Configuration N8N et tests
