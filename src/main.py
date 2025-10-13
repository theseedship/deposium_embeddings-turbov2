from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from model2vec import StaticModel
import torch
import torch.quantization as quant
import copy
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

    # Load Qwen3-Embedding-0.6B for RERANKING (NOT embeddings!)
    # 4-bit quantization with bitsandbytes for Railway deployment
    logger.info("Loading Qwen3-Embedding-0.6B for reranking (4-bit quantized)...")
    try:
        # Check if running on CPU (Railway) or GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {device}")

        # For CPU deployment (Railway), load without quantization initially
        # Quantization will be added later when we have proper CPU support
        if device == "cpu":
            logger.info("Loading Qwen3 on CPU (no quantization for now)...")
            models["qwen3-rerank"] = SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B",
                trust_remote_code=True,
                device="cpu"
            )
            logger.info("✅ Qwen3-Embedding-0.6B loaded on CPU! (RERANKING MODEL)")
        else:
            # GPU: use 4-bit quantization
            logger.info("Loading Qwen3 on GPU with 4-bit quantization...")
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            models["qwen3-rerank"] = SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B",
                trust_remote_code=True,
                device="cuda",
                model_kwargs={"quantization_config": quantization_config}
            )
            logger.info("✅ Qwen3-Embedding-0.6B loaded with 4-bit quantization! (GPU)")
    except Exception as e:
        logger.warning(f"⚠️ Could not load Qwen3-Embedding: {e}")

    logger.info("🚀 All models ready!")

# Request/Response models
class EmbedRequest(BaseModel):
    model: str = "gemma-768d"
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
        "gemma-768d": "⚡ Gemma-768D Model2Vec (PRIMARY) - 500-700x faster! Quality: 0.659 | Semantic: 0.730 | Multilingual: 0.690",
        "int8": "C10X/int8 (256D) - lightweight reranker",
        "qwen3-rerank": "🎯 Qwen3-Embedding-0.6B (596M params, MTEB: 64.33) - Full reranker with 4-bit quantization",
    }

    return {
        "service": "Deposium Embeddings - Multi-Model Evaluation",
        "status": "running",
        "version": "6.1.0",
        "models": model_info,
        "recommended": "gemma-768d (WINNER: Best quality + speed + multilingual)",
        "quality_metrics": {
            "gemma-768d": {
                "overall": 0.6587,
                "semantic_similarity": 0.7302,
                "topic_clustering": 0.5558,
                "multilingual": 0.6903,
                "dimensions": 768,
                "speed": "500-700x faster than full Gemma",
                "use_case": "Fast embeddings"
            },
            "qwen3-rerank": {
                "mteb_score": 64.33,
                "retrieval_score": 76.17,
                "parameters": "596M",
                "dimensions": "up to 1024",
                "quantization": "4-bit (GPU) or FP32 (CPU)",
                "use_case": "High-quality reranking"
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
        },
        {
            "name": "qwen3-rerank",
            "size": 600000000,  # ~600MB (596M params)
            "digest": "qwen3-rerank-4bit",
            "modified_at": "2025-10-14T00:00:00Z",
            "details": "🎯 Qwen3-Embedding-0.6B (596M params) - Full reranker with 4-bit quantization (MTEB: 64.33)"
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

@app.post("/api/rerank")
async def rerank_documents(request: RerankRequest):
    """
    Rerank documents by relevance to a query

    Supports:
    - qwen3-rerank: Full Qwen3-Embedding-0.6B with 4-bit quantization (MTEB: 64.33)
    - int8: Lightweight C10X/int8 reranker (256D)

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

        # For SentenceTransformer models (qwen3-rerank), use native encode
        if isinstance(selected_model, SentenceTransformer):
            # Encode query and documents
            query_emb = selected_model.encode(request.query, convert_to_tensor=True)
            doc_embs = selected_model.encode(request.documents, convert_to_tensor=True)

            # Calculate cosine similarity scores
            scores = cos_sim(query_emb, doc_embs)[0].cpu().tolist()
        else:
            # For Model2Vec models (int8), use standard encode
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
