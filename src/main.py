from fastapi import FastAPI, HTTPException, File, UploadFile, Header, Depends
from pydantic import BaseModel
from typing import List, Optional
from model2vec import StaticModel
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from pathlib import Path
import logging
import os

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


# ============================================================================
# Authentication
# ============================================================================

async def verify_api_key(x_api_key: str = Header(..., description="API Key for authentication")):
    """
    Verify API key from X-API-Key header.

    Environment variable: EMBEDDINGS_API_KEY

    If EMBEDDINGS_API_KEY is not set, authentication is disabled (dev mode).
    """
    expected_key = os.getenv("EMBEDDINGS_API_KEY")

    # Dev mode: allow if no key configured
    if not expected_key:
        logger.warning("⚠️ EMBEDDINGS_API_KEY not configured - authentication disabled!")
        return "dev-mode"

    # Validate key
    if x_api_key != expected_key:
        logger.warning(f"Invalid API key attempt: {x_api_key[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    return x_api_key

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
    logger.info("  • Priority system: High-priority models stay in VRAM")
    logger.info("  • Auto-unloading: Frees VRAM when limit exceeded")
    
    logger.info("\nAvailable Models:")
    logger.info("  • qwen25-1024d: Instruction-aware embeddings (priority: 10)")
    logger.info("  • gemma-768d: Multilingual embeddings (priority: 5)")
    logger.info("  • qwen3-rerank: Document reranking (priority: 8)")
    logger.info("  • embeddinggemma-300m: Full-size embeddings (priority: 2)")
    logger.info("  • qwen3-embed: Full-size embeddings (priority: 7)")
    
    # Optionally preload high-priority models
    # Disabled by default to minimize startup VRAM usage
    # model_manager.preload_priority_models()
    
    logger.info("✅ Model Manager initialized! Models will load on first use.")

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
async def create_embedding(request: EmbedRequest, api_key: str = Depends(verify_api_key)):
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
async def create_embedding_alt(request: EmbedRequest, api_key: str = Depends(verify_api_key)):
    """Alternative endpoint (some clients use /api/embeddings)"""
    return await create_embedding(request, api_key)

@app.post("/api/rerank")
async def rerank_documents(request: RerankRequest, api_key: str = Depends(verify_api_key)):
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


@app.post("/api/classify/file")
async def classify_document_file(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    """
    Classify document complexity from uploaded file (multipart/form-data).

    **Usage:**
    ```bash
    curl -X POST http://localhost:11435/api/classify/file \\
      -H "X-API-Key: YOUR_API_KEY" \\
      -F "file=@document.jpg"
    ```

    **Returns:**
    - class_name: "LOW" (simple OCR) or "HIGH" (VLM reasoning)
    - confidence: 0.0-1.0
    - probabilities: {"LOW": float, "HIGH": float}
    - routing_decision: str (routing recommendation)
    - latency_ms: float

    **Use case:** Route LOW complexity to OCR (~100ms), HIGH to VLM (~2000ms)
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Expected image/*"
            )

        # Get classifier (lazy loading)
        classifier = get_classifier()

        # Predict
        result = await classifier.predict_from_file(file)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/classify/base64")
async def classify_document_base64(request: ClassifyRequest, api_key: str = Depends(verify_api_key)):
    """
    Classify document complexity from base64 encoded image (application/json).

    **Usage:**
    ```bash
    curl -X POST http://localhost:11435/api/classify/base64 \\
      -H "X-API-Key: YOUR_API_KEY" \\
      -H "Content-Type: application/json" \\
      -d '{"image":"data:image/jpeg;base64,/9j/4AAQ..."}'
    ```

    **Returns:**
    - class_name: "LOW" (simple OCR) or "HIGH" (VLM reasoning)
    - confidence: 0.0-1.0
    - probabilities: {"LOW": float, "HIGH": float}
    - routing_decision: str (routing recommendation)
    - latency_ms: float

    **Use case:** Route LOW complexity to OCR (~100ms), HIGH to VLM (~2000ms)
    """
    try:
        # Validate input
        if request.image is None:
            raise HTTPException(
                status_code=400,
                detail="Missing 'image' field in JSON body"
            )

        # Get classifier (lazy loading)
        classifier = get_classifier()

        # Predict
        result = await classifier.predict_from_base64(request.image)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
