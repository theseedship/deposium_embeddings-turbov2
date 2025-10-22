from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
from model2vec import StaticModel
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from pathlib import Path
import logging
import os
import asyncio

# Import classifier module
from .classifier import get_classifier, ClassifyRequest
# Import model manager
from .model_manager import get_model_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Deposium Embeddings - Qwen25 Instruction-Aware + Full-Size Models",
    description="🔥 NEW: Qwen25-1024D Instruction-Aware (65MB, quality: 0.84) + Gemma-768D Model2Vec + Full-size models (EmbeddingGemma-300M, Qwen3-0.6B) + FP32 Reranking",
    version="10.0.0"
)

# Initialize model manager
model_manager = None

# Periodic cleanup task
async def periodic_cleanup():
    """Periodically cleanup inactive models to save memory."""
    logger.info(f"Starting periodic cleanup task (checks every 30s, timeout from AUTO_UNLOAD_MODELS_TIME={os.getenv('AUTO_UNLOAD_MODELS_TIME', '180')}s)")
    while True:
        await asyncio.sleep(30)  # Check every 30 seconds
        if model_manager:
            try:
                model_manager.cleanup_inactive_models()  # Uses AUTO_UNLOAD_MODELS_TIME env var
            except Exception as e:
                logger.error(f"Error during periodic cleanup: {e}")

@app.on_event("startup")
async def initialize_models():
    global model_manager

    # Initialize model manager with lazy loading
    logger.info("=" * 80)
    logger.info("🚀 Initializing Model Manager with Dynamic VRAM Management")
    logger.info("=" * 80)
    
    model_manager = get_model_manager()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    if device == "cuda":
        used_mb, free_mb = model_manager.get_vram_usage_mb()
        total_mb = used_mb + free_mb
        logger.info(f"GPU Memory: {used_mb}MB used, {free_mb}MB free (Total: {total_mb}MB)")
        logger.info(f"VRAM Limit: {model_manager.max_vram_mb}MB (keeps 1GB margin)")
    
    logger.info("\nModel Loading Strategy:")
    logger.info("  • Lazy loading: Models loaded only when needed")
    logger.info(f"  • Auto-unloading: After {os.getenv('AUTO_UNLOAD_MODELS_TIME', '180')} seconds of inactivity")
    logger.info("  • Memory optimization: All models have equal priority")
    
    logger.info("\nAvailable Models:")
    logger.info("  • qwen25-1024d: Instruction-aware embeddings")
    logger.info("  • gemma-768d: Multilingual embeddings")
    logger.info("  • qwen3-rerank: Document reranking")
    logger.info("  • qwen3-embed: Full-size embeddings")
    logger.info("  • vl-classifier: Visual document complexity classification")
    
    # Start periodic cleanup task
    asyncio.create_task(periodic_cleanup())
    logger.info("  • Started automatic cleanup task (checks every 30s)")
    
    logger.info("✅ Model Manager initialized! Models will load on demand and unload after 3 minutes of inactivity.")

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
    global model_manager
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    status = model_manager.get_status()
    return {
        "status": "healthy",
        "models_loaded": list(status.get("loaded_models", {})),
        "vram_used_mb": status.get("vram_used_mb", 0),
        "vram_free_mb": status.get("vram_free_mb", 0)
    }

@app.get("/api/status")
async def get_status():
    """Get detailed model manager status"""
    global model_manager
    if not model_manager:
        return {"error": "Model manager not initialized"}
    return model_manager.get_status()

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
    global model_manager
    
    # Validate model selection
    available_models = model_manager.configs.keys()
    if request.model not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found. Available: {list(available_models)}"
        )

    try:
        # Handle both string and list inputs
        texts = [request.input] if isinstance(request.input, str) else request.input

        # Get model (lazy loading)
        selected_model = model_manager.get_model(request.model)

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
    global model_manager
    
    # Validate model selection
    available_models = model_manager.configs.keys()
    if request.model not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found. Available: {list(available_models)}"
        )

    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    try:
        # Get model (lazy loading)
        selected_model = model_manager.get_model(request.model)

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


@app.post("/api/classify")
async def classify_document(
    request: ClassifyRequest = None,
    file: UploadFile = File(None)
):
    """
    Classify document complexity for intelligent routing.

    Supports two input methods:
    1. JSON with base64 image: {"image": "data:image/jpeg;base64,..."}
    2. Multipart file upload: file=@document.jpg

    Returns:
    - class_name: "LOW" (simple OCR) or "HIGH" (VLM reasoning)
    - confidence: 0.0-1.0
    - probabilities: {"LOW": float, "HIGH": float}
    - routing_decision: str (routing recommendation)
    - latency_ms: float

    Use case: Route LOW complexity to OCR (~100ms), HIGH to VLM (~2000ms)
    """
    try:
        # Get classifier (lazy loading)
        classifier = get_classifier()

        # Validate input
        if request is None and file is None:
            raise HTTPException(
                status_code=400,
                detail="Either 'image' in JSON body or 'file' in multipart required"
            )

        # Route to appropriate prediction method
        if file is not None:
            # File upload (multipart/form-data)
            if not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {file.content_type}. Expected image/*"
                )
            result = await classifier.predict_from_file(file)
        elif request is not None and request.image is not None:
            # Base64 image (JSON)
            result = await classifier.predict_from_base64(request.image)
        else:
            raise HTTPException(
                status_code=400,
                detail="No image provided. Send 'image' (base64) in JSON or 'file' in multipart"
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
