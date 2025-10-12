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
    title="Deposium Embeddings - Qwen3-Turbo + TurboX.v2 + Gemma",
    description="Multi-model embeddings: Qwen3-Turbo (710x faster!) + Model2Vec + EmbeddingGemma (high quality)",
    version="5.0.0"
)

# Load models at startup
models = {}

@app.on_event("startup")
async def load_models():
    global models

    # Load Qwen3-Turbo (ULTRA FAST - 710x faster than gemma-int8!)
    logger.info("Loading Qwen3-Turbo model (256D, 710x faster)...")
    models["qwen3-turbo"] = StaticModel.from_pretrained("Pringled/m2v-Qwen3-Embedding-0.6B")
    logger.info("✅ Qwen3-Turbo loaded! (Quality: 0.665, Speed: 710x faster)")

    logger.info("Loading TurboX.v2 model (1024D)...")
    models["turbov2"] = StaticModel.from_pretrained("C10X/Qwen3-Embedding-TurboX.v2")
    logger.info("✅ TurboX.v2 loaded!")

    logger.info("Loading int8 reranker model (256D)...")
    models["int8"] = StaticModel.from_pretrained("C10X/int8")
    logger.info("✅ int8 loaded!")

    # Load EmbeddingGemma variants (baseline + INT8 optimized)
    try:
        logger.info("Loading EmbeddingGemma-300m variants...")

        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load base model in float32 (required for INT8 quantization base)
        logger.info(f"Loading base model on {device}...")
        model_base = SentenceTransformer("google/embeddinggemma-300m")
        model_base = model_base.to(device)

        # Variant 1: gemma (float16 for GPU, float32 for CPU to avoid zero bug)
        if torch.cuda.is_available():
            logger.info("Creating gemma variant (float16 GPU-optimized)...")
            model_fp16 = SentenceTransformer("google/embeddinggemma-300m")
            model_fp16 = model_fp16.half().to(device)
            models["gemma"] = model_fp16
            logger.info("✅ gemma (float16) loaded - ~587MB")
        else:
            # On CPU, use float32 to avoid zero bug
            logger.info("Creating gemma variant (float32 CPU-compatible)...")
            models["gemma"] = model_base
            logger.info("✅ gemma (float32) loaded - ~1.2GB")

        # Variant 2: gemma-int8 (INT8 quantized for 3x speedup on CPU)
        logger.info("Creating gemma-int8 variant (INT8 quantized)...")

        # Create a CPU copy for INT8 quantization (INT8 dynamic quant only works on CPU)
        model_int8 = copy.deepcopy(model_base)
        model_int8 = model_int8.to("cpu")  # Move to CPU before quantization

        # Apply dynamic INT8 quantization (quantizes weights only, CPU-only)
        model_int8 = quant.quantize_dynamic(
            model_int8,
            {torch.nn.Linear},  # Quantize all Linear layers
            dtype=torch.qint8
        )

        models["gemma-int8"] = model_int8
        logger.info("✅ gemma-int8 (INT8) loaded - ~300MB, 3x faster CPU")

        logger.info(f"✅ EmbeddingGemma variants ready!")
        logger.info(f"   Device: {device}")
        logger.info(f"   Max seq length: {model_base.max_seq_length} tokens")
        logger.info(f"   Embedding dim: {model_base.get_sentence_embedding_dimension()}D")
        logger.info(f"   Variants:")
        logger.info(f"     - gemma: {'float16 GPU (~587MB)' if torch.cuda.is_available() else 'float32 CPU (~1.2GB)'}")
        logger.info(f"     - gemma-int8: INT8 quantized (~300MB, 3x faster)")
        logger.info(f"   MTEB Score: 0.80 Spearman (BIOSSES: 0.83, STSBenchmark: 0.88)")

    except Exception as e:
        logger.warning(f"⚠️  Failed to load EmbeddingGemma variants: {e}")

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
    model_info = {
        "qwen3-turbo": "⚡ Qwen3-Turbo (256D) - ULTRA FAST! 710x faster than gemma-int8 (Quality: 0.665)",
        "turbov2": "C10X/Qwen3-Embedding-TurboX.v2 (1024D) - ultra-fast",
        "int8": "C10X/int8 (256D) - compact",
    }
    if "gemma" in models:
        model_info["gemma"] = "EmbeddingGemma-300m (768D, 2048 tokens) - high quality (MTEB: 0.80)"
    if "gemma-int8" in models:
        model_info["gemma-int8"] = "EmbeddingGemma-300m INT8 (768D, 2048 tokens) - 3x faster CPU"

    return {
        "service": "Deposium Embeddings - Qwen3-Turbo + TurboX.v2 + EmbeddingGemma",
        "status": "running",
        "version": "5.0.0",
        "models": model_info,
        "recommended": "qwen3-turbo (710x faster, Railway-optimized)"
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
            "name": "qwen3-turbo",
            "size": 200000000,  # ~200MB
            "digest": "qwen3-turbo-256d-m2v",
            "modified_at": "2025-10-12T00:00:00Z",
            "details": "⚡ Qwen3-Turbo Model2Vec (256D) - 710x FASTER! Quality: 0.665, Railway-optimized"
        },
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

    # Add EmbeddingGemma variants if loaded
    if "gemma" in models:
        model_list.append({
            "name": "gemma",
            "size": 587000000,  # ~587MB (float16) or ~1200000000 (float32)
            "digest": "gemma-768d-float16",
            "modified_at": "2025-10-12T00:00:00Z",
            "details": "EmbeddingGemma-300m baseline (768D, 2048 tokens, MTEB: 0.80)"
        })

    if "gemma-int8" in models:
        model_list.append({
            "name": "gemma-int8",
            "size": 300000000,  # ~300MB (INT8)
            "digest": "gemma-768d-int8",
            "modified_at": "2025-10-12T00:00:00Z",
            "details": "EmbeddingGemma-300m INT8 quantized (768D, 2048 tokens, 3x faster CPU)"
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
        if request.model in ["gemma", "gemma-int8"]:
            # EmbeddingGemma variants use SentenceTransformer
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

            # Check for NaN/Inf/Zero values (float16 edge case on CPU) - fallback to float32
            has_nan_or_inf = np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings))
            is_all_zeros = np.allclose(embeddings, 0.0, atol=1e-6)  # Check if all values are near zero

            if has_nan_or_inf or is_all_zeros:
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
            # Model2Vec models (qwen3-turbo, turbov2, int8)
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
