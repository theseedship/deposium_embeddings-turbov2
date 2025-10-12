# HuggingFace Token Configuration

EmbeddingGemma-300m est un modèle gated qui nécessite un token HuggingFace pour l'accès.

## Étapes de configuration

### 1. Créer un token HuggingFace

1. Allez sur https://huggingface.co/settings/tokens
2. Cliquez sur "New token"
3. Nom: `deposium-embeddings-token` (ou autre)
4. Type: **Read** (lecture seule suffit)
5. Copiez le token généré

### 2. Accepter les termes d'utilisation

Visitez https://huggingface.co/google/embeddinggemma-300m et cliquez sur "Agree and access repository"

### 3. Configurer le token localement

#### Pour Docker local:

Éditez `/home/nico/code_source/tss/deposium_fullstack/deposium-local/.env`:

```bash
HF_TOKEN=votre_token_ici
```

Puis rebuild et redémarrez le service:

```bash
cd /home/nico/code_source/tss/deposium_fullstack/deposium-local
docker-compose up -d --build embeddings-turbov2
```

#### Pour développement Python (venv):

Éditez `/home/nico/code_source/tss/deposium_embeddings-turbov2/.env`:

```bash
HF_TOKEN=votre_token_ici
```

Puis démarrez l'API:

```bash
cd ~/code_source/tss/deposium_embeddings-turbov2
source venv/bin/activate
HF_TOKEN=$(cat .env | grep HF_TOKEN | cut -d'=' -f2) uvicorn src.main:app --host 0.0.0.0 --port 11435
```

### 4. Configurer le token sur Railway

1. Allez dans le dashboard Railway du projet
2. Sélectionnez le service `deposium_embeddings-turbov2`
3. Variables → Add Variable
4. Nom: `HF_TOKEN`
5. Valeur: votre token HuggingFace

Railway redéploiera automatiquement.

## Vérification

### Test local:

```bash
curl -X POST http://localhost:11435/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma", "input": "test de connexion"}'
```

### Test Docker:

```bash
curl -X POST http://deposium-embeddings-turbov2:11435/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma", "input": "test de connexion"}'
```

## Sécurité

⚠️ **IMPORTANT**:
- Ne commitez JAMAIS le fichier `.env` avec votre token
- Le `.env` est dans `.gitignore`
- Si un token est exposé publiquement, révoquez-le immédiatement sur https://huggingface.co/settings/tokens

## Logs à surveiller

### Succès:
```
INFO:src.main:Loading EmbeddingGemma-300m baseline (768D, float16, 2048 tokens)...
INFO:src.main:✅ EmbeddingGemma loaded!
```

### Échec d'authentification:
```
401 Client Error: Repository gated. You are trying to access a gated repo.
```

→ Solution: Vérifiez que le token est configuré et que vous avez accepté les termes.

### NaN/Fallback float32:
```
WARNING:src.main:Float16 NaN detected, retrying with float32 for N texts
INFO:src.main:Float32 fallback successful for N texts
```

→ Normal: Le système fallback automatiquement en float32 pour certains textes problématiques.
