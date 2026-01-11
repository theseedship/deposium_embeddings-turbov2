from fastapi import FastAPI, HTTPException, File, UploadFile, Header, Depends, Request
from pydantic import BaseModel, Field
from typing import List, Optional
from model2vec import StaticModel
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# MXBAI Reranker import
try:
    from mxbai_rerank import MxbaiRerankV2
    MXBAI_RERANK_AVAILABLE = True
except ImportError:
    MxbaiRerankV2 = None
    MXBAI_RERANK_AVAILABLE = False
from pathlib import Path
import logging
import os

# Import classifier module
from .classifier import get_classifier, ClassifyRequest
# Import model manager
from .model_manager import get_model_manager
# Import benchmark runner
from .benchmarks import OpenBenchRunner, BenchmarkCategory, get_openbench_runner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Deposium Embeddings - M2V-BGE-M3 + BGE-M3 ONNX",
    description="🔥 M2V-BGE-M3-1024D (distilled from BGE-M3, ~3x more energy efficient) + BGE-M3-ONNX INT8 (CPU high quality) + Qwen3 Reranking + VL Classifier",
    version="11.0.0"
)

# Initialize model manager
model_manager = None


# ============================================================================
# Authentication
# ============================================================================

async def verify_api_key(request: Request):
    """
    Verify API key from either Authorization Bearer or X-API-Key header.

    Supports two authentication formats:
    - Authorization: Bearer <token> (Ollama standard - N8N compatible)
    - X-API-Key: <token> (Custom header - backward compatible)

    Special handling:
    - Railway internal network (*.railway.internal): Authentication bypassed
    - External requests: API key required

    Environment variable: EMBEDDINGS_API_KEY

    If EMBEDDINGS_API_KEY is not set, authentication is disabled (dev mode).
    """
    # Check if request is from Railway internal network
    host = request.headers.get("host", "")
    if ".railway.internal" in host:
        logger.info(f"🔓 Railway internal network request from {host} - authentication bypassed")
        return "railway-internal"

    # For external requests, check API key configuration
    expected_key = os.getenv("EMBEDDINGS_API_KEY")

    # Dev mode: allow if no key configured
    if not expected_key:
        logger.warning("⚠️ EMBEDDINGS_API_KEY not configured - authentication disabled!")
        return "dev-mode"

    # Extract token from headers (try both formats)
    token = None
    auth_method = None

    # Try Authorization: Bearer header
    authorization = request.headers.get("authorization")
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]  # Remove "Bearer " prefix
        auth_method = "Bearer"

    # Try X-API-Key header if Bearer not found
    if not token:
        x_api_key = request.headers.get("x-api-key")
        if x_api_key:
            token = x_api_key
            auth_method = "X-API-Key"

    # Validate token
    if not token or token != expected_key:
        logger.warning(f"❌ Invalid API key attempt from {host} (method: {auth_method or 'none'})")
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Use 'Authorization: Bearer <token>' or 'X-API-Key: <token>' header",
            headers={"WWW-Authenticate": "Bearer"}
        )

    logger.info(f"✅ Authentication successful from {host} (method: {auth_method})")
    return token

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
    logger.info("  • m2v-bge-m3-1024d: Distilled BGE-M3 embeddings (priority: 10) - 3x more energy efficient")
    logger.info("  • bge-m3-onnx: BGE-M3 ONNX INT8 for CPU (priority: 8) - high quality")
    logger.info("  • gemma-768d: Legacy multilingual embeddings (priority: 5)")
    logger.info("  • qwen3-rerank: Document reranking (priority: 8)")
    logger.info("  • vl-classifier: Document complexity classifier (ONNX, standalone)")
    
    # Optionally preload high-priority models
    # Disabled by default to minimize startup VRAM usage
    # model_manager.preload_priority_models()
    
    # Start background cleanup task
    import asyncio
    async def model_cleanup_loop():
        """Background task to cleanup inactive models."""
        logger.info("🧹 Starting model cleanup background task")
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                if model_manager:
                    model_manager.cleanup_inactive_models()
            except Exception as e:
                logger.error(f"Error in model cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    asyncio.create_task(model_cleanup_loop())
    
    logger.info("✅ Model Manager initialized! Models will load on first use.")

# Default model names from environment variables
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "m2v-bge-m3-1024d")
DEFAULT_RERANK_MODEL = os.getenv("DEFAULT_RERANK_MODEL", "qwen3-rerank")

# Request/Response models
class EmbedRequest(BaseModel):
    model: str = Field(default=DEFAULT_EMBEDDING_MODEL, description="Model to use for embeddings")
    input: str | List[str]

class EmbedResponse(BaseModel):
    model: str
    embeddings: List[List[float]]

class RerankRequest(BaseModel):
    model: str = Field(default=DEFAULT_RERANK_MODEL, description="Model to use for reranking")
    query: str
    documents: List[str]
    top_k: Optional[int] = None  # Return all by default


class VisionRequest(BaseModel):
    """Request for vision-language model inference"""
    model: str = Field(default="lfm25-vl", description="Vision-language model to use")
    image: str = Field(..., description="Base64 encoded image (with or without data URI prefix)")
    prompt: str = Field(default="Extract all text from this document.", description="Prompt for the model")
    max_tokens: Optional[int] = Field(default=512, description="Maximum tokens to generate")


class VisionResponse(BaseModel):
    """Response from vision-language model"""
    model: str
    response: str
    latency_ms: float

class RerankResponse(BaseModel):
    model: str
    results: List[dict]  # [{"index": 0, "document": "...", "relevance_score": 0.95}, ...]

@app.get("/")
async def root():
    model_info = {
        "m2v-bge-m3-1024d": "🔥 M2V-BGE-M3 (PRIMARY) - Distilled from BGE-M3! 3x more energy efficient | MTEB: 0.47 | 1024D",
        "bge-m3-onnx": "⚡ BGE-M3 ONNX INT8 (CPU) - High quality embeddings | MTEB: ~0.60 | 1024D",
        "gemma-768d": "📚 Gemma-768D Model2Vec (LEGACY) - Multilingual | MTEB: 0.55",
        "qwen3-rerank": "🏆 Qwen3 FP32 Reranking - FASTEST + BEST PRECISION (242ms for 3 docs!)",
        "vl-classifier": "🎯 Document Complexity Classifier - ResNet18 ONNX INT8 (93% accuracy, ~10ms)",
    }

    return {
        "service": "Deposium Embeddings - M2V-BGE-M3 + ONNX + Rerank + Classifier",
        "status": "running",
        "version": "11.0.0",
        "models": model_info,
        "recommended": "m2v-bge-m3-1024d for fast embeddings, bge-m3-onnx for quality, qwen3-rerank for reranking",
        "quality_metrics": {
            "m2v-bge-m3-1024d": {
                "overall_mteb": 0.47,
                "sts": 0.58,
                "classification": 0.66,
                "clustering": 0.28,
                "dimensions": 1024,
                "size_mb": 21,
                "params": "1.54B distilled to 21MB",
                "energy_efficiency": "3x more efficient than SentenceTransformers",
                "throughput": "14,171 texts/s (vs 2,587 for MiniLM)",
                "texts_per_wh": 3191024,
                "use_case": "Primary model - fast RAG, bulk processing, energy-efficient deployments"
            },
            "bge-m3-onnx": {
                "overall_mteb": 0.60,
                "dimensions": 1024,
                "size_mb": 150,
                "quantization": "INT8 ONNX",
                "device": "CPU optimized",
                "use_case": "High quality CPU embeddings when GPU not available"
            },
            "gemma-768d": {
                "overall_mteb": 0.55,
                "semantic_similarity": 0.59,
                "multilingual": 0.74,
                "dimensions": 768,
                "size_mb": 400,
                "params": "~50M",
                "use_case": "Legacy - multilingual support"
            },
            "qwen3-rerank": {
                "mteb_score": 64.33,
                "retrieval_score": 76.17,
                "params": "596M",
                "speed": "242ms for 3 docs (Railway vCPU)",
                "precision": "BEST (0.126 separation Paris-London)",
                "quantization": "FP32 (environment optimizations make it fastest!)",
                "use_case": "Reranking - best speed + precision on Railway vCPU"
            },
            "vl-classifier": {
                "architecture": "ResNet18 ONNX INT8",
                "accuracy": 0.93,
                "high_recall": 1.00,
                "params": "11M",
                "latency": "10-17ms on CPU",
                "size_mb": 11,
                "classes": ["LOW", "HIGH"],
                "use_case": "Document routing - simple (OCR) vs complex (VLM)"
            }
        },
        "energy_benchmark": {
            "note": "Custom benchmark using CodeCarbon (Model2Vec not compatible with AIEnergyScore)",
            "m2v-bge-m3-1024d": {"texts_per_wh": 3191024, "throughput": "14,171 texts/s"},
            "m2v-qwen3-1024d": {"texts_per_wh": 3047706, "throughput": "14,056 texts/s"},
            "all-MiniLM-L6-v2": {"texts_per_wh": 1129181, "throughput": "2,587 texts/s"},
            "comparison": "Model2Vec is ~3x more energy efficient than SentenceTransformers"
        }
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
            "name": "m2v-bge-m3-1024d",
            "size": 21000000,  # ~21MB
            "digest": "m2v-bge-m3-1024d-distilled",
            "modified_at": "2025-12-05T00:00:00Z",
            "details": "🔥 M2V-BGE-M3 (PRIMARY) - Distilled from BGE-M3 | MTEB: 0.47 | 3x energy efficient | 14k texts/s"
        },
        {
            "name": "bge-m3-onnx",
            "size": 150000000,  # ~150MB
            "digest": "bge-m3-onnx-int8",
            "modified_at": "2025-12-05T00:00:00Z",
            "details": "⚡ BGE-M3 ONNX INT8 (CPU) - High quality embeddings | MTEB: ~0.60 | 1024D"
        },
        {
            "name": "gemma-768d",
            "size": 400000000,  # ~400MB
            "digest": "gemma-768d-m2v-deposium",
            "modified_at": "2025-10-13T00:00:00Z",
            "details": "📚 Gemma-768D (LEGACY) - Multilingual | MTEB: 0.55"
        },
        {
            "name": "qwen3-rerank",
            "size": 600000000,  # ~600MB (FP32)
            "digest": "qwen3-rerank-fp32-optimized",
            "modified_at": "2025-10-14T00:00:00Z",
            "details": "🏆 Qwen3 FP32 Reranking - FASTEST (242ms) + BEST PRECISION on Railway vCPU!"
        },
        {
            "name": "vl-classifier",
            "size": 11000000,  # ~11MB
            "digest": "resnet18-onnx-int8",
            "modified_at": "2025-10-22T00:00:00Z",
            "details": "🎯 Document Complexity Classifier - 93% accuracy, ~10ms latency"
        },
        {
            "name": "mxbai-embed-2d",
            "size": 800000000,  # ~800MB
            "digest": "mxbai-embed-2d-large-v1",
            "modified_at": "2026-01-10T00:00:00Z",
            "details": "🎯 MXBAI-Embed-2D (24 layers) - 2D Matryoshka SOTA English | 1024D"
        },
        {
            "name": "mxbai-embed-2d-fast",
            "size": 400000000,  # ~400MB
            "digest": "mxbai-embed-2d-large-v1-12layers",
            "modified_at": "2026-01-10T00:00:00Z",
            "details": "⚡ MXBAI-Embed-2D Fast (12 layers) - ~2x speedup | 768D"
        },
        {
            "name": "mxbai-embed-2d-turbo",
            "size": 250000000,  # ~250MB
            "digest": "mxbai-embed-2d-large-v1-6layers",
            "modified_at": "2026-01-10T00:00:00Z",
            "details": "🚀 MXBAI-Embed-2D Turbo (6 layers) - ~4x speedup | 512D"
        },
        {
            "name": "mxbai-rerank-v2",
            "size": 250000000,  # ~250MB with 4-bit quantization
            "digest": "mxbai-rerank-base-v2-4bit",
            "modified_at": "2026-01-11T00:00:00Z",
            "details": "🏆 MXBAI-Rerank-V2 SOTA cross-encoder | BEIR 55.57 | 100+ languages | 4-bit NF4"
        },
        {
            "name": "lfm25-vl",
            "size": 3200000000,  # ~3.2GB
            "digest": "lfm25-vl-1.6b",
            "modified_at": "2026-01-11T00:00:00Z",
            "details": "👁️ LFM2.5-VL-1.6B Vision-Language | Document OCR | Edge-first CPU design | 1.6B params"
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

        # Handle 2D Matryoshka dimension truncation
        # Models like mxbai-embed-2d-fast/turbo have _truncate_dims attribute
        truncate_dims = getattr(selected_model, '_truncate_dims', None)
        if truncate_dims:
            # Truncate embeddings to specified dimensions
            embeddings = embeddings[:, :truncate_dims]

        embeddings_list = [emb.tolist() for emb in embeddings]

        # Log dimensions for debugging
        dims = len(embeddings_list[0]) if embeddings_list else 0
        truncation_info = f" (truncated to {truncate_dims}D)" if truncate_dims else ""
        logger.info(f"Generated {len(embeddings_list)} embeddings with {dims}D using {request.model}{truncation_info}")

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
    Rerank documents by relevance to a query.

    **Models:**
    - mxbai-rerank-v2: SOTA cross-encoder (BEIR 55.57, 100+ languages) - RECOMMENDED
    - qwen3-rerank: Bi-encoder with cosine similarity (faster, lower quality)
    - Embedding models: Can also use for reranking via cosine similarity

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

        # Check model type for appropriate reranking strategy
        model_config = model_manager.configs.get(request.model)

        # MXBAI Reranker (true cross-encoder - SOTA quality)
        if model_config and model_config.type == "mxbai_reranker":
            if not MXBAI_RERANK_AVAILABLE:
                raise HTTPException(
                    status_code=500,
                    detail="mxbai-rerank library not installed. Install with: pip install mxbai-rerank"
                )

            # Use native cross-encoder reranking
            top_k = request.top_k if request.top_k else len(request.documents)
            ranked_results = selected_model.rank(
                request.query,
                request.documents,
                return_documents=True,
                top_k=top_k
            )

            # Convert to our response format
            # mxbai-rerank returns RankResult objects with .index, .text, .score attributes
            results = [
                {
                    "index": item.index,
                    "document": item.text,
                    "relevance_score": float(item.score)
                }
                for i, item in enumerate(ranked_results)
            ]

            logger.info(f"Reranked {len(request.documents)} documents with {request.model} (cross-encoder), top score: {results[0]['relevance_score']:.4f}")

        # SentenceTransformer models (bi-encoder with cosine similarity)
        elif isinstance(selected_model, SentenceTransformer):
            # Encode query and documents
            query_emb = selected_model.encode(request.query, convert_to_tensor=True)
            doc_embs = selected_model.encode(request.documents, convert_to_tensor=True)

            # Calculate cosine similarity scores
            scores = cos_sim(query_emb, doc_embs)[0].cpu().tolist()

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

            logger.info(f"Reranked {len(request.documents)} documents with {request.model} (bi-encoder), top score: {results[0]['relevance_score']:.4f}")

        # Model2Vec or other embedding models
        else:
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

            logger.info(f"Reranked {len(request.documents)} documents with {request.model} (embedding), top score: {results[0]['relevance_score']:.4f}")

        return {
            "model": request.model,
            "results": results
        }

    except Exception as e:
        logger.error(f"Reranking error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/classify/file")
async def classify_document_file(
    file: UploadFile = File(...),
    model: str = "vl-classifier",
    api_key: str = Depends(verify_api_key)
):
    """
    Classify document complexity from uploaded file (multipart/form-data).

    **Models:**
    - vl-classifier: ResNet18 ONNX (fast, ~10ms)
    - lfm25-vl: LFM2.5-VL-1.6B VLM (accurate, ~10-15s)

    **Usage:**
    ```bash
    curl -X POST http://localhost:11435/api/classify/file \\
      -H "X-API-Key: YOUR_API_KEY" \\
      -F "file=@document.jpg" \\
      -F "model=vl-classifier"
    ```

    **Returns:**
    - class_name: "LOW" (simple OCR) or "HIGH" (VLM reasoning)
    - confidence: 0.0-1.0
    - probabilities: {"LOW": float, "HIGH": float}
    - routing_decision: str (routing recommendation)
    - latency_ms: float

    **Use case:** Route LOW complexity to OCR (~100ms), HIGH to VLM (~2000ms)
    """
    global model_manager
    import io
    import time
    from PIL import Image

    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Expected image/*"
            )

        # Use LFM2.5-VL for classification if requested
        if model == "lfm25-vl":
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            vlm_model, vlm_processor = model_manager.get_model("lfm25-vl")

            # Classification prompt
            prompt = """Analyze this document image and classify its complexity:
- LOW: Simple text, single column, no tables, easy to OCR
- HIGH: Complex layout, tables, forms, multiple columns, needs VLM reasoning

Answer with ONLY one word: LOW or HIGH"""

            conversation = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}]

            inputs = vlm_processor.apply_chat_template(
                conversation, add_generation_prompt=True,
                return_tensors="pt", return_dict=True, tokenize=True
            ).to(vlm_model.device)

            start_time = time.time()
            outputs = vlm_model.generate(**inputs, max_new_tokens=10)
            latency_ms = (time.time() - start_time) * 1000

            response = vlm_processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # Parse response
            is_high = "HIGH" in response.upper()
            class_name = "HIGH" if is_high else "LOW"

            return {
                "class_name": class_name,
                "confidence": 0.85,  # VLM doesn't provide exact confidence
                "probabilities": {"LOW": 0.15 if is_high else 0.85, "HIGH": 0.85 if is_high else 0.15},
                "routing_decision": f"Route to {'VLM reasoning' if is_high else 'fast OCR'}",
                "latency_ms": round(latency_ms, 2),
                "model": "lfm25-vl"
            }

        # Default: Use ResNet18 classifier
        classifier = get_classifier()
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

    **Models:**
    - vl-classifier: ResNet18 ONNX (fast, ~10ms) - default
    - lfm25-vl: LFM2.5-VL-1.6B VLM (accurate, ~10-15s)

    **Usage:**
    ```bash
    curl -X POST http://localhost:11435/api/classify/base64 \\
      -H "X-API-Key: YOUR_API_KEY" \\
      -H "Content-Type: application/json" \\
      -d '{"image":"data:image/jpeg;base64,/9j/4AAQ...", "model":"lfm25-vl"}'
    ```

    **Returns:**
    - class_name: "LOW" (simple OCR) or "HIGH" (VLM reasoning)
    - confidence: 0.0-1.0
    - probabilities: {"LOW": float, "HIGH": float}
    - routing_decision: str (routing recommendation)
    - latency_ms: float

    **Use case:** Route LOW complexity to OCR (~100ms), HIGH to VLM (~2000ms)
    """
    global model_manager
    import base64
    import io
    import time
    from PIL import Image

    try:
        # Validate input
        if request.image is None:
            raise HTTPException(
                status_code=400,
                detail="Missing 'image' field in JSON body"
            )

        # Use LFM2.5-VL for classification if requested
        if request.model == "lfm25-vl":
            # Decode base64 image
            image_data = request.image
            if image_data.startswith("data:"):
                image_data = image_data.split(",", 1)[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            vlm_model, vlm_processor = model_manager.get_model("lfm25-vl")

            # Classification prompt
            prompt = """Analyze this document image and classify its complexity:
- LOW: Simple text, single column, no tables, easy to OCR
- HIGH: Complex layout, tables, forms, multiple columns, needs VLM reasoning

Answer with ONLY one word: LOW or HIGH"""

            conversation = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}]

            inputs = vlm_processor.apply_chat_template(
                conversation, add_generation_prompt=True,
                return_tensors="pt", return_dict=True, tokenize=True
            ).to(vlm_model.device)

            start_time = time.time()
            outputs = vlm_model.generate(**inputs, max_new_tokens=10)
            latency_ms = (time.time() - start_time) * 1000

            response = vlm_processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # Parse response
            is_high = "HIGH" in response.upper()
            class_name = "HIGH" if is_high else "LOW"

            return {
                "class_name": class_name,
                "confidence": 0.85,  # VLM doesn't provide exact confidence
                "probabilities": {"LOW": 0.15 if is_high else 0.85, "HIGH": 0.85 if is_high else 0.15},
                "routing_decision": f"Route to {'VLM reasoning' if is_high else 'fast OCR'}",
                "latency_ms": round(latency_ms, 2),
                "model": "lfm25-vl"
            }

        # Default: Use ResNet18 classifier
        classifier = get_classifier()
        result = await classifier.predict_from_base64(request.image)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Vision-Language API (Document OCR with LFM2.5-VL)
# ============================================================================

@app.post("/api/vision", response_model=VisionResponse)
async def process_vision(request: VisionRequest, api_key: str = Depends(verify_api_key)):
    """
    Process an image with a vision-language model for OCR and document understanding.

    **Models:**
    - lfm25-vl: LFM2.5-VL-1.6B (excellent OCR, edge-first design)

    **Usage:**
    ```bash
    curl -X POST http://localhost:11436/api/vision \\
      -H "X-API-Key: YOUR_API_KEY" \\
      -H "Content-Type: application/json" \\
      -d '{
        "model": "lfm25-vl",
        "image": "data:image/png;base64,iVBORw0KGgo...",
        "prompt": "Extract all text from this document."
      }'
    ```

    **Prompts examples:**
    - "Extract all text from this document." (OCR)
    - "Is this document SIMPLE or COMPLEX? Answer with just one word." (Classification)
    - "Summarize the main content in 2-3 sentences." (Summary)
    - "What type of document is this?" (Document type detection)

    **Returns:**
    - model: Model used
    - response: Generated text response
    - latency_ms: Processing time in milliseconds
    """
    global model_manager
    import base64
    import io
    import time
    from PIL import Image

    # Validate model
    if request.model not in model_manager.configs:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found. Available vision models: lfm25-vl"
        )

    model_config = model_manager.configs[request.model]
    if model_config.type != "vision_language":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is not a vision-language model"
        )

    try:
        # Decode base64 image
        image_data = request.image
        if image_data.startswith("data:"):
            # Remove data URI prefix (e.g., "data:image/png;base64,")
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Get model (lazy loading) - returns (model, processor) tuple
        vlm_model, vlm_processor = model_manager.get_model(request.model)

        # Prepare conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": request.prompt},
                ],
            },
        ]

        # Process inputs
        inputs = vlm_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(vlm_model.device)

        # Generate response
        start_time = time.time()
        outputs = vlm_model.generate(**inputs, max_new_tokens=request.max_tokens or 512)
        latency_ms = (time.time() - start_time) * 1000

        # Decode response
        response_text = vlm_processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract just the assistant's response (remove the prompt echo)
        if "assistant" in response_text.lower():
            # Try to extract text after "assistant" marker
            parts = response_text.split("assistant")
            if len(parts) > 1:
                response_text = parts[-1].strip()

        logger.info(f"Vision processed with {request.model} in {latency_ms:.0f}ms")

        return VisionResponse(
            model=request.model,
            response=response_text,
            latency_ms=round(latency_ms, 2)
        )

    except Exception as e:
        logger.error(f"Vision processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vision/file")
async def process_vision_file(
    file: UploadFile = File(...),
    prompt: str = "Extract all text from this document.",
    model: str = "lfm25-vl",
    max_tokens: int = 512,
    api_key: str = Depends(verify_api_key)
):
    """
    Process an uploaded image file with a vision-language model.

    **Usage:**
    ```bash
    curl -X POST http://localhost:11436/api/vision/file \\
      -H "X-API-Key: YOUR_API_KEY" \\
      -F "file=@document.png" \\
      -F "prompt=Extract all text from this document."
    ```

    **Returns:**
    - model: Model used
    - response: Generated text response
    - latency_ms: Processing time in milliseconds
    """
    global model_manager
    import base64
    import io
    import time
    from PIL import Image

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected image/*"
        )

    # Validate model
    if model not in model_manager.configs:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' not found. Available vision models: lfm25-vl"
        )

    model_config = model_manager.configs[model]
    if model_config.type != "vision_language":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not a vision-language model"
        )

    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Get model (lazy loading) - returns (model, processor) tuple
        vlm_model, vlm_processor = model_manager.get_model(model)

        # Prepare conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Process inputs
        inputs = vlm_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(vlm_model.device)

        # Generate response
        start_time = time.time()
        outputs = vlm_model.generate(**inputs, max_new_tokens=max_tokens)
        latency_ms = (time.time() - start_time) * 1000

        # Decode response
        response_text = vlm_processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract just the assistant's response
        if "assistant" in response_text.lower():
            parts = response_text.split("assistant")
            if len(parts) > 1:
                response_text = parts[-1].strip()

        logger.info(f"Vision file processed with {model} in {latency_ms:.0f}ms")

        return {
            "model": model,
            "response": response_text,
            "latency_ms": round(latency_ms, 2)
        }

    except Exception as e:
        logger.error(f"Vision file processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Benchmarks API (OpenBench Integration)
# ============================================================================

class BenchmarkRequest(BaseModel):
    """Request for running a benchmark"""
    category: str = Field(
        default="search",
        description="Benchmark category: knowledge, coding, math, reasoning, cybersecurity, search"
    )
    provider: Optional[str] = Field(
        default="groq",
        description="LLM provider (groq, openai, anthropic)"
    )
    model: Optional[str] = Field(
        default="llama-3.1-8b-instant",
        description="Model name"
    )
    sample_limit: Optional[int] = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum samples to evaluate"
    )
    custom_corpus: Optional[List[dict]] = Field(
        default=None,
        description="Custom corpus data for search benchmarks"
    )


@app.get("/api/benchmarks")
async def list_benchmarks(api_key: str = Depends(verify_api_key)):
    """
    List available benchmark categories and providers.

    Returns information about:
    - Available categories (knowledge, coding, math, reasoning, cybersecurity, search)
    - Supported providers
    - Default configuration
    """
    try:
        runner = get_openbench_runner()
        return await runner.list_available_benchmarks()
    except Exception as e:
        logger.error(f"Benchmark list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/benchmarks/run")
async def run_benchmark(request: BenchmarkRequest, api_key: str = Depends(verify_api_key)):
    """
    Run a benchmark evaluation.

    **Categories:**
    - `knowledge`: General knowledge (MMLU, TriviaQA)
    - `coding`: Code generation (HumanEval, MBPP)
    - `math`: Mathematical reasoning (GSM8K, MATH)
    - `reasoning`: Logic and deduction (ARC, HellaSwag)
    - `cybersecurity`: Security tasks
    - `search`: Retrieval quality (custom corpus supported)

    **For custom corpus evaluation:**
    ```json
    {
        "category": "search",
        "custom_corpus": [
            {
                "query": "What is machine learning?",
                "relevant_docs": ["doc1_content", "doc2_content"]
            }
        ]
    }
    ```

    **Returns:**
    - score: Overall benchmark score (0-1)
    - metrics: Detailed metrics
    - samples_evaluated: Number of samples tested
    - duration_seconds: Execution time
    """
    try:
        runner = get_openbench_runner()
        result = await runner.run_benchmark(
            category=request.category,
            provider=request.provider,
            model=request.model,
            custom_dataset=request.custom_corpus,
            sample_limit=request.sample_limit or 100
        )
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Benchmark run error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/benchmarks/corpus-eval")
async def evaluate_corpus(
    corpus_data: List[dict],
    provider: Optional[str] = "groq",
    model: Optional[str] = "llama-3.1-8b-instant",
    sample_limit: Optional[int] = 100,
    api_key: str = Depends(verify_api_key)
):
    """
    Evaluate a Deposium corpus for retrieval quality.

    **Input format:**
    ```json
    [
        {
            "query": "Search query",
            "relevant_docs": ["Expected relevant document content..."],
            "context": "Optional context"
        }
    ]
    ```

    **Returns:**
    - Retrieval precision metrics
    - Per-query scores
    - Aggregate quality score
    """
    try:
        runner = get_openbench_runner()
        result = await runner.run_benchmark(
            category="search",
            provider=provider,
            model=model,
            custom_dataset=corpus_data,
            sample_limit=sample_limit
        )
        return result.to_dict()
    except Exception as e:
        logger.error(f"Corpus evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
