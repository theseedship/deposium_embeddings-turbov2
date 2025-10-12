from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from model2vec import StaticModel
import torch
from transformers import AutoTokenizer
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Deposium Embeddings - TurboX.v2 + int8 + LEAF",
    description="Multi-model embeddings: Model2Vec (fast) + LEAF (accurate)",
    version="3.0.0"
)

# Load models at startup
models = {}

@app.on_event("startup")
async def load_models():
    global models
    logger.info("Loading TurboX.v2 model (1024D)...")
    models["turbov2"] = StaticModel.from_pretrained("C10X/Qwen3-Embedding-TurboX.v2")
    logger.info("✅ TurboX.v2 loaded!")

    logger.info("Loading int8 reranker model (256D)...")
    models["int8"] = StaticModel.from_pretrained("C10X/int8")
    logger.info("✅ int8 loaded!")

    # Load LEAF model from Hugging Face
    try:
        logger.info("Loading LEAF model from Hugging Face (768D, INT8 quantized)...")
        from huggingface_hub import hf_hub_download

        # Download model files from HF
        model_path = hf_hub_download(
            repo_id="tss-deposium/gemma300-leaf-embeddings-test",
            filename="model_quantized.pt",
            cache_dir="/tmp/hf_cache"
        )

        # Load model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        models["leaf"] = checkpoint['model']
        models["leaf"].eval()

        # Load tokenizer from HF
        tokenizer = AutoTokenizer.from_pretrained(
            "tss-deposium/gemma300-leaf-embeddings-test",
            cache_dir="/tmp/hf_cache"
        )
        models["leaf"].set_tokenizer(tokenizer)

        logger.info("✅ LEAF loaded! (695 texts/s on CPU)")
    except Exception as e:
        logger.warning(f"⚠️  Failed to load LEAF from HF: {e}")

    logger.info("🚀 All models ready!")

# Request/Response models
class EmbedRequest(BaseModel):
    model: str = "turbov2"
    input: str | List[str]

class EmbedResponse(BaseModel):
    model: str
    embeddings: List[List[float]]

@app.get("/")
async def root():
    return {
        "service": "Deposium Embeddings - TurboX.v2 + int8 + LEAF",
        "status": "running",
        "models": {
            "turbov2": "C10X/Qwen3-Embedding-TurboX.v2 (1024D) - ultra-fast",
            "int8": "C10X/int8 (256D) - compact",
            "leaf": "LEAF INT8 (768D) - accurate, 695 texts/s CPU"
        }
    }

@app.get("/health")
async def health():
    if not models or len(models) == 0:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "status": "healthy",
        "models_loaded": list(models.keys())
    }

@app.get("/api/tags")
async def list_models():
    """Ollama-compatible endpoint to list models"""
    model_list = [
        {
            "name": "turbov2",
            "size": 30000000,  # ~30MB
            "digest": "turbov2-1024d",
            "modified_at": "2025-10-09T00:00:00Z",
            "details": "C10X/Qwen3-Embedding-TurboX.v2 (1024 dimensions)"
        },
        {
            "name": "int8",
            "size": 30000000,  # ~30MB
            "digest": "int8-256d",
            "modified_at": "2025-10-09T00:00:00Z",
            "details": "C10X/int8 (256 dimensions)"
        }
    ]

    # Add LEAF if loaded
    if "leaf" in models:
        model_list.append({
            "name": "leaf",
            "size": 441000000,  # ~441MB
            "digest": "leaf-768d-int8",
            "modified_at": "2025-10-12T00:00:00Z",
            "details": "LEAF INT8 quantized (768 dimensions, 695 texts/s CPU)"
        })

    return {"models": model_list}

@app.post("/api/embed")
async def create_embedding(request: EmbedRequest):
    """Ollama-compatible embedding endpoint with multi-model support"""
    # Validate model selection
    if request.model not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found. Available: {list(models.keys())}"
        )

    try:
        # Handle both string and list inputs
        texts = [request.input] if isinstance(request.input, str) else request.input

        # Select the appropriate model
        selected_model = models[request.model]

        # Generate embeddings based on model type
        if request.model == "leaf":
            # LEAF uses PyTorch (different interface)
            with torch.no_grad():
                embeddings = selected_model.encode(texts, device='cpu', normalize=True)
            embeddings_list = embeddings.tolist()
        else:
            # Model2Vec models (turbov2, int8)
            embeddings = selected_model.encode(texts, show_progress_bar=False)
            embeddings_list = [emb.tolist() for emb in embeddings]

        # Log dimensions for debugging
        dims = len(embeddings_list[0]) if embeddings_list else 0
        logger.info(f"Generated {len(embeddings_list)} embeddings with {dims}D using {request.model}")

        return {
            "model": request.model,
            "embeddings": embeddings_list
        }

    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/embeddings")
async def create_embedding_alt(request: EmbedRequest):
    """Alternative endpoint (some clients use /api/embeddings)"""
    return await create_embedding(request)
