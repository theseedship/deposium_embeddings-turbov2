from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from model2vec import StaticModel
import torch
import torch.quantization as quant
import copy
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Deposium Embeddings - Gemma-768D Model2Vec + Reranker",
    description="Ultra-fast embeddings: Gemma-768D Model2Vec (Quality: 0.659, Multilingual: 0.690, 500-700x faster!)",
    version="6.0.0"
)

# Load models at startup
models = {}

@app.on_event("startup")
async def load_models():
    global models

    # Load Gemma-768D Model2Vec (PRIMARY MODEL - Winner!)
    logger.info("Loading Gemma-768D Model2Vec (PRIMARY)...")
    logger.info("  Quality: 0.6587 | Semantic: 0.7302 | Clustering: 0.5558 | Multilingual: 0.6903")

    # Try loading from local path first, then from Hugging Face
    gemma_768d_local = Path("models/gemma-deposium-768d")
    if gemma_768d_local.exists():
        logger.info("Loading Gemma-768D from local path...")
        models["gemma-768d"] = StaticModel.from_pretrained(str(gemma_768d_local))
        logger.info("✅ Gemma-768D Model2Vec loaded from local! (768D, 500-700x faster)")
    else:
        logger.info("Local model not found, downloading from Hugging Face...")
        try:
            models["gemma-768d"] = StaticModel.from_pretrained("tss-deposium/gemma-deposium-768d")
            logger.info("✅ Gemma-768D Model2Vec downloaded from HF! (768D, 500-700x faster)")
        except Exception as e:
            logger.error(f"❌ Failed to load Gemma-768D: {e}")
            raise RuntimeError("Primary model Gemma-768D not found!")

    # Load reranker (int8)
    logger.info("Loading int8 reranker model (256D)...")
    models["int8"] = StaticModel.from_pretrained("C10X/int8")
    logger.info("✅ int8 reranker loaded!")

    # Optional: Load Qwen3-256D for comparison (if needed)
    # logger.info("Loading Qwen3-256D for comparison...")
    # models["qwen3-256d"] = StaticModel.from_pretrained("Pringled/m2v-Qwen3-Embedding-0.6B")
    # logger.info("✅ Qwen3-256D loaded (Quality: 0.555, Multilingual: 0.316 - POOR)")

    logger.info("🚀 All models ready!")

# Request/Response models
class EmbedRequest(BaseModel):
    model: str = "gemma-768d"
    input: str | List[str]

class EmbedResponse(BaseModel):
    model: str
    embeddings: List[List[float]]

@app.get("/")
async def root():
    model_info = {
        "gemma-768d": "⚡ Gemma-768D Model2Vec (PRIMARY) - 500-700x faster! Quality: 0.659 | Semantic: 0.730 | Multilingual: 0.690",
        "int8": "C10X/int8 (256D) - reranker model",
    }

    return {
        "service": "Deposium Embeddings - Gemma-768D Model2Vec + Reranker",
        "status": "running",
        "version": "6.0.0",
        "models": model_info,
        "recommended": "gemma-768d (WINNER: Best quality + speed + multilingual)",
        "quality_metrics": {
            "gemma-768d": {
                "overall": 0.6587,
                "semantic_similarity": 0.7302,
                "topic_clustering": 0.5558,
                "multilingual": 0.6903,
                "dimensions": 768,
                "speed": "500-700x faster than full Gemma"
            }
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
            "name": "gemma-768d",
            "size": 400000000,  # ~400MB
            "digest": "gemma-768d-m2v-deposium",
            "modified_at": "2025-10-13T00:00:00Z",
            "details": "⚡ Gemma-768D Model2Vec (PRIMARY) - Quality: 0.659 | Multilingual: 0.690 | 500-700x FASTER!"
        },
        {
            "name": "int8",
            "size": 30000000,  # ~30MB
            "digest": "int8-256d-reranker",
            "modified_at": "2025-10-09T00:00:00Z",
            "details": "C10X/int8 (256D) - Reranker model"
        }
    ]

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

        # Generate embeddings - all models now use Model2Vec
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
