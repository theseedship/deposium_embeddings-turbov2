from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from model2vec import StaticModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Deposium Embeddings - TurboX.v2 + int8",
    description="Ultra-fast static embeddings service with dual models (1024D & 256D)",
    version="2.0.0"
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
        "service": "Deposium Embeddings - TurboX.v2 + int8",
        "status": "running",
        "models": {
            "turbov2": "C10X/Qwen3-Embedding-TurboX.v2 (1024D)",
            "int8": "C10X/int8 (256D)"
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
    return {
        "models": [
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
    }

@app.post("/api/embed")
async def create_embedding(request: EmbedRequest):
    """Ollama-compatible embedding endpoint with dual model support"""
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

        # Generate embeddings
        embeddings = selected_model.encode(texts, show_progress_bar=False)

        # Convert to list format
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
