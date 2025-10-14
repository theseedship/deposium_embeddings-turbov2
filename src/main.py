from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from model2vec import StaticModel
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from pathlib import Path
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Deposium Embeddings - Qwen25 Instruction-Aware + Full-Size Models",
    description="🔥 NEW: Qwen25-1024D Instruction-Aware (65MB, quality: 0.84) + Gemma-768D Model2Vec + Full-size models (EmbeddingGemma-300M, Qwen3-0.6B) + FP32 Reranking",
    version="10.0.0"
)

# Load models at startup
models = {}

@app.on_event("startup")
async def load_models():
    global models

    # Load Qwen25-1024D Model2Vec (PRIMARY MODEL - NEW CHAMPION! 🏆)
    logger.info("=" * 80)
    logger.info("🔥 Loading Qwen25-1024D Model2Vec (PRIMARY - INSTRUCTION-AWARE)")
    logger.info("=" * 80)
    logger.info("  Overall Quality: 0.841 (+52% vs Gemma-768D)")
    logger.info("  Instruction-Aware: 0.953 (UNIQUE capability)")
    logger.info("  Semantic: 0.950 | Code: 0.864 | Conversational: 0.846")
    logger.info("  Size: 65MB (6x smaller than Gemma-768D)")
    logger.info("  Speed: 500-1000x faster than full LLM")

    # Try loading from local_models (in Docker image) first, then from Hugging Face
    # Note: /app/models is Railway volume (HuggingFace cache), /app/local_models is in image
    qwen25_local = Path("/app/local_models/qwen25-deposium-1024d")
    qwen25_fallback = Path("models/qwen25-deposium-1024d")  # For local dev

    if qwen25_local.exists():
        logger.info("Loading Qwen25-1024D from Docker image (/app/local_models)...")
        models["qwen25-1024d"] = StaticModel.from_pretrained(str(qwen25_local))
        logger.info("✅ Qwen25-1024D Model2Vec loaded from image! (1024D, instruction-aware)")
    elif qwen25_fallback.exists():
        logger.info("Loading Qwen25-1024D from local dev path...")
        models["qwen25-1024d"] = StaticModel.from_pretrained(str(qwen25_fallback))
        logger.info("✅ Qwen25-1024D Model2Vec loaded from local dev! (1024D, instruction-aware)")
    else:
        logger.info("Local model not found, downloading from Hugging Face...")
        try:
            models["qwen25-1024d"] = StaticModel.from_pretrained("tss-deposium/qwen25-deposium-1024d")
            logger.info("✅ Qwen25-1024D Model2Vec downloaded from HF! (1024D, instruction-aware)")
        except Exception as e:
            logger.error(f"❌ Failed to load Qwen25-1024D: {e}")
            raise RuntimeError("Primary model Qwen25-1024D not found!")

    # Load Gemma-768D Model2Vec (SECONDARY - still available)
    logger.info("\nLoading Gemma-768D Model2Vec (SECONDARY)...")
    logger.info("  Quality: 0.551 | Multilingual: 0.737")

    gemma_768d_local = Path("/app/local_models/gemma-deposium-768d")
    gemma_768d_fallback = Path("models/gemma-deposium-768d")  # For local dev

    if gemma_768d_local.exists():
        logger.info("Loading Gemma-768D from Docker image (/app/local_models)...")
        models["gemma-768d"] = StaticModel.from_pretrained(str(gemma_768d_local))
        logger.info("✅ Gemma-768D Model2Vec loaded from image! (768D, 500-700x faster)")
    elif gemma_768d_fallback.exists():
        logger.info("Loading Gemma-768D from local dev path...")
        models["gemma-768d"] = StaticModel.from_pretrained(str(gemma_768d_fallback))
        logger.info("✅ Gemma-768D Model2Vec loaded from local dev! (768D, 500-700x faster)")
    else:
        logger.info("Local model not found, downloading from Hugging Face...")
        try:
            models["gemma-768d"] = StaticModel.from_pretrained("tss-deposium/gemma-deposium-768d")
            logger.info("✅ Gemma-768D Model2Vec downloaded from HF! (768D, 500-700x faster)")
        except Exception as e:
            logger.warning(f"⚠️ Could not load Gemma-768D: {e}")

    # Load full-size embedding models (for comparison with distilled versions)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load EmbeddingGemma-300M (full-size Gemma embeddings)
    logger.info("Loading EmbeddingGemma-300M (full-size embeddings)...")
    try:
        models["embeddinggemma-300m"] = SentenceTransformer(
            "google/embeddinggemma-300m",
            trust_remote_code=True,
            device=device
        )
        logger.info("✅ EmbeddingGemma-300M loaded! (300M params, 768D)")
    except Exception as e:
        logger.warning(f"⚠️ Could not load EmbeddingGemma-300M: {e}")

    # Load Qwen3-Embedding-0.6B (for both embeddings AND reranking)
    logger.info("Loading Qwen3-Embedding-0.6B (embeddings + reranking)...")
    try:
        models["qwen3-embed"] = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            trust_remote_code=True,
            device=device
        )
        logger.info("✅ Qwen3-Embedding-0.6B loaded! (600M params, 1024D, MTEB: 64.33)")

        # Also use for reranking (FP32 = best speed + precision on Railway vCPU!)
        models["qwen3-rerank"] = models["qwen3-embed"]
        logger.info("✅ Qwen3 also configured for reranking (242ms, best precision!)")
    except Exception as e:
        logger.warning(f"⚠️ Could not load Qwen3-Embedding: {e}")

    logger.info("🚀 All models ready!")

# Request/Response models
class EmbedRequest(BaseModel):
    model: str = "qwen25-1024d"  # Changed to qwen25-1024d as primary
    input: str | List[str]

class EmbedResponse(BaseModel):
    model: str
    embeddings: List[List[float]]

class RerankRequest(BaseModel):
    model: str = "qwen3-rerank"
    query: str
    documents: List[str]
    top_k: Optional[int] = None  # Return all by default

class RerankResponse(BaseModel):
    model: str
    results: List[dict]  # [{"index": 0, "document": "...", "relevance_score": 0.95}, ...]

@app.get("/")
async def root():
    model_info = {
        "qwen25-1024d": "🔥 Qwen25-1024D (PRIMARY) - Instruction-Aware! Quality: 0.841 | Instruction: 0.953 | 65MB",
        "gemma-768d": "⚡ Gemma-768D Model2Vec (SECONDARY) - 500-700x faster! Quality: 0.551 | Multilingual: 0.737",
        "embeddinggemma-300m": "🎯 EmbeddingGemma-300M (FULL-SIZE) - 300M params, 768D, high quality embeddings",
        "qwen3-embed": "🚀 Qwen3-Embedding-0.6B (FULL-SIZE) - 600M params, 1024D, MTEB: 64.33",
        "qwen3-rerank": "🏆 Qwen3 FP32 Reranking - FASTEST + BEST PRECISION (242ms for 3 docs!)",
    }

    return {
        "service": "Deposium Embeddings - Qwen25 Instruction-Aware + Full-Size Models",
        "status": "running",
        "version": "10.0.0",
        "models": model_info,
        "recommended": "qwen25-1024d for instruction-aware + quality, gemma-768d for multilingual, qwen3-rerank for reranking",
        "quality_metrics": {
            "qwen25-1024d": {
                "overall": 0.841,
                "instruction_awareness": 0.953,
                "semantic_similarity": 0.950,
                "code_understanding": 0.864,
                "conversational": 0.846,
                "multilingual": 0.434,
                "dimensions": 1024,
                "size_mb": 65,
                "params": "1.54B distilled",
                "speed": "500-1000x faster than full LLM",
                "unique_capability": "Instruction-aware embeddings (ONLY model with this)",
                "use_case": "Primary model - instruction-aware RAG, Q&A, code search"
            },
            "gemma-768d": {
                "overall": 0.551,
                "semantic_similarity": 0.591,
                "multilingual": 0.737,
                "conversational": 0.159,
                "dimensions": 768,
                "size_mb": 400,
                "params": "~50M",
                "speed": "500-700x faster than full Gemma",
                "use_case": "Secondary - multilingual support"
            },
            "embeddinggemma-300m": {
                "params": "300M",
                "dimensions": 768,
                "use_case": "Full-size Gemma embeddings (higher quality than distilled)"
            },
            "qwen3-embed": {
                "mteb_score": 64.33,
                "retrieval_score": 76.17,
                "params": "596M",
                "dimensions": 1024,
                "use_case": "Full-size embeddings (best quality)"
            },
            "qwen3-rerank": {
                "mteb_score": 64.33,
                "retrieval_score": 76.17,
                "params": "596M",
                "speed": "242ms for 3 docs (Railway vCPU)",
                "precision": "BEST (0.126 separation Paris-London)",
                "quantization": "FP32 (environment optimizations make it fastest!)",
                "use_case": "Reranking - best speed + precision on Railway vCPU"
            }
        },
        "optimization_note": "FP32 models benefit massively from environment optimizations (OMP_NUM_THREADS, jemalloc, KMP_AFFINITY)"
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
            "name": "qwen25-1024d",
            "size": 65000000,  # ~65MB
            "digest": "qwen25-1024d-m2v-instruction-aware",
            "modified_at": "2025-10-14T00:00:00Z",
            "details": "🔥 Qwen25-1024D (PRIMARY) - Instruction-Aware! Quality: 0.841 | 65MB | 500-1000x FASTER"
        },
        {
            "name": "gemma-768d",
            "size": 400000000,  # ~400MB (corrected size)
            "digest": "gemma-768d-m2v-deposium",
            "modified_at": "2025-10-13T00:00:00Z",
            "details": "⚡ Gemma-768D (SECONDARY) - Quality: 0.551 | Multilingual: 0.737"
        },
        {
            "name": "embeddinggemma-300m",
            "size": 300000000,  # ~300MB
            "digest": "embeddinggemma-300m-full",
            "modified_at": "2025-10-14T00:00:00Z",
            "details": "🎯 EmbeddingGemma-300M (FULL-SIZE) - 300M params, 768D, high quality"
        },
        {
            "name": "qwen3-embed",
            "size": 600000000,  # ~600MB
            "digest": "qwen3-embed-fp32",
            "modified_at": "2025-10-14T00:00:00Z",
            "details": "🚀 Qwen3-0.6B (FULL-SIZE) - 600M params, 1024D, MTEB: 64.33"
        },
        {
            "name": "qwen3-rerank",
            "size": 600000000,  # ~600MB (FP32)
            "digest": "qwen3-rerank-fp32-optimized",
            "modified_at": "2025-10-14T00:00:00Z",
            "details": "🏆 Qwen3 FP32 Reranking - FASTEST (242ms) + BEST PRECISION on Railway vCPU!"
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

        # Generate embeddings (Model2Vec or SentenceTransformer)
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

@app.post("/api/rerank")
async def rerank_documents(request: RerankRequest):
    """
    Rerank documents by relevance to a query using FP32 models

    - qwen3-rerank: FP32 Qwen3-0.6B (242ms, BEST precision on Railway vCPU!)
    - Can also use embedding models for reranking

    Returns documents sorted by relevance score (highest first)
    """
    # Validate model selection
    if request.model not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found. Available: {list(models.keys())}"
        )

    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    try:
        selected_model = models[request.model]

        # For SentenceTransformer models (qwen3-rerank, embeddinggemma-300m, qwen3-embed)
        if isinstance(selected_model, SentenceTransformer):
            # Encode query and documents
            query_emb = selected_model.encode(request.query, convert_to_tensor=True)
            doc_embs = selected_model.encode(request.documents, convert_to_tensor=True)

            # Calculate cosine similarity scores
            scores = cos_sim(query_emb, doc_embs)[0].cpu().tolist()
        else:
            # For Model2Vec models (gemma-768d), use standard encode
            query_emb = selected_model.encode([request.query], show_progress_bar=False)[0]
            doc_embs = selected_model.encode(request.documents, show_progress_bar=False)

            # Calculate cosine similarity manually
            import numpy as np
            scores = [
                np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
                for doc_emb in doc_embs
            ]

        # Create results with original indices
        results = [
            {
                "index": i,
                "document": doc,
                "relevance_score": float(score)
            }
            for i, (doc, score) in enumerate(zip(request.documents, scores))
        ]

        # Sort by relevance (highest first)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Apply top_k if specified
        if request.top_k:
            results = results[:request.top_k]

        logger.info(f"Reranked {len(request.documents)} documents with {request.model}, top score: {results[0]['relevance_score']:.4f}")

        return {
            "model": request.model,
            "results": results
        }

    except Exception as e:
        logger.error(f"Reranking error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
