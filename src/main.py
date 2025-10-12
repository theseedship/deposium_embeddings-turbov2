from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from model2vec import StaticModel
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Deposium Embeddings - TurboX.v2 + int8 + Gemma",
    description="Multi-model embeddings: Model2Vec (fast) + EmbeddingGemma (high quality, 2048 tokens)",
    version="4.0.0"
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

    # Load EmbeddingGemma baseline (replaces LEAF)
    try:
        logger.info("Loading EmbeddingGemma-300m baseline (768D, float16, 2048 tokens)...")

        # Load from HuggingFace with float16 quantization
        model = SentenceTransformer("google/embeddinggemma-300m")
        model = model.half()  # Quantize to float16 (50% size reduction)

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        models["gemma"] = model

        logger.info(f"✅ EmbeddingGemma loaded!")
        logger.info(f"   Device: {device}")
        logger.info(f"   Max seq length: {model.max_seq_length} tokens")
        logger.info(f"   Embedding dim: {model.get_sentence_embedding_dimension()}D")
        logger.info(f"   Quantization: float16 (~587MB)")
        logger.info(f"   MTEB Score: 0.80 Spearman (BIOSSES: 0.83, STSBenchmark: 0.88)")

    except Exception as e:
        logger.warning(f"⚠️  Failed to load EmbeddingGemma: {e}")

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
        "service": "Deposium Embeddings - TurboX.v2 + int8 + EmbeddingGemma",
        "status": "running",
        "version": "4.0.0",
        "models": {
            "turbov2": "C10X/Qwen3-Embedding-TurboX.v2 (1024D) - ultra-fast",
            "int8": "C10X/int8 (256D) - compact",
            "gemma": "EmbeddingGemma-300m (768D, 2048 tokens) - high quality (MTEB: 0.80)"
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

    # Add EmbeddingGemma if loaded
    if "gemma" in models:
        model_list.append({
            "name": "gemma",
            "size": 587000000,  # ~587MB (float16)
            "digest": "gemma-768d-float16",
            "modified_at": "2025-10-12T00:00:00Z",
            "details": "EmbeddingGemma-300m baseline (768D, 2048 tokens, MTEB: 0.80)"
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
        if request.model == "gemma":
            # EmbeddingGemma uses SentenceTransformer
            embeddings = selected_model.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=True,  # Model handles normalization correctly
                batch_size=8,  # Optimized from benchmarks
                convert_to_numpy=True  # Ensure numpy array output
            )
            # Convert float16 to float32 for JSON serialization
            import numpy as np
            embeddings = np.array(embeddings, dtype=np.float32)

            # Check for NaN/Inf values (float16 edge case) - fallback to float32
            if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                logger.warning(f"Float16 NaN detected, retrying with float32 for {len(texts)} texts")
                # Reload model in float32 and retry
                model_fp32 = SentenceTransformer("google/embeddinggemma-300m")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_fp32 = model_fp32.to(device)

                embeddings = model_fp32.encode(
                    texts,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=8,
                    convert_to_numpy=True
                )
                embeddings = np.array(embeddings, dtype=np.float32)
                logger.info(f"Float32 fallback successful for {len(texts)} texts")

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
