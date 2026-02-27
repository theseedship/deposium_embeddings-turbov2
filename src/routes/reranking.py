"""Document reranking routes."""
import gc
import logging
import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from .. import shared
from ..shared import run_sync
from ..schemas.requests import RerankRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/rerank")
async def rerank_documents(request: RerankRequest, api_key: str = Depends(shared.verify_api_key)):
    """
    Rerank documents by relevance to a query.

    **Models:**
    - bge-reranker-v2-m3: ONNX INT8 cross-encoder (MIRACL FR 59.6, CPU optimized, ~569MB)
    - mxbai-rerank-v2: Cross-encoder (BEIR 55.57, 100+ languages)
    - mxbai-rerank-xsmall: Lighter cross-encoder (~40% faster, 278M params)
    - Embedding models: Can also use for reranking via cosine similarity

    Returns documents sorted by relevance score (highest first)
    """

    # Validate model selection
    available_models = shared.model_manager.configs.keys()
    if request.model not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found. Available: {list(available_models)}"
        )

    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    try:
        # Get model (lazy loading)
        selected_model = shared.model_manager.get_model(request.model)

        # Check model type for appropriate reranking strategy
        model_config = shared.model_manager.configs.get(request.model)

        # ONNX Reranker (BGE-reranker-v2-m3 ONNX INT8 - CPU optimized cross-encoder)
        if model_config and model_config.type == "onnx_reranker":
            top_k = request.top_k if request.top_k else len(request.documents)

            ranked_results = await run_sync(
                selected_model.rank,
                request.query,
                request.documents,
                top_k=top_k,
            )

            # OnnxRerankerModel.rank() returns list of dicts with index, score, document
            results = [
                {
                    "index": item["index"],
                    "document": item["document"],
                    "relevance_score": item["score"]
                }
                for item in ranked_results
            ]

            logger.info(f"Reranked {len(request.documents)} documents with {request.model} (ONNX cross-encoder), top score: {results[0]['relevance_score']:.4f}")

        # MXBAI Reranker (true cross-encoder - SOTA quality)
        # Supports both V1 (DeBERTa) and V2 (Qwen2) architectures
        elif model_config and model_config.type in ("mxbai_reranker", "mxbai_reranker_v1"):
            if not shared.MXBAI_RERANK_AVAILABLE:
                raise HTTPException(
                    status_code=500,
                    detail="mxbai-rerank library not installed. Install with: pip install mxbai-rerank"
                )

            # Use native cross-encoder reranking (CPU/GPU-heavy, offload to thread pool)
            top_k = request.top_k if request.top_k else len(request.documents)

            ranked_results = await run_sync(
                selected_model.rank,
                request.query,
                request.documents,
                return_documents=True,
                top_k=top_k,
            )

            # Convert to our response format
            # mxbai-rerank returns RankResult objects with .index, .document, .score attributes
            results = [
                {
                    "index": item.index,
                    "document": item.document,
                    "relevance_score": float(item.score)
                }
                for i, item in enumerate(ranked_results)
            ]

            logger.info(f"Reranked {len(request.documents)} documents with {request.model} (cross-encoder), top score: {results[0]['relevance_score']:.4f}")

        # SentenceTransformer models (bi-encoder with cosine similarity)
        elif isinstance(selected_model, SentenceTransformer):
            # Encode query and documents (GPU-heavy, offload to thread pool)
            def _rerank_biencoder():
                with torch.inference_mode():
                    query_emb = selected_model.encode(request.query, convert_to_tensor=True)
                    doc_embs = selected_model.encode(request.documents, convert_to_tensor=True)
                    scores = cos_sim(query_emb, doc_embs)[0].cpu().tolist()

                # Free CUDA tensors immediately
                del query_emb, doc_embs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return scores

            scores = await run_sync(_rerank_biencoder)

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
            def _rerank_embedding():
                query_emb = selected_model.encode([request.query], show_progress_bar=False)[0]
                doc_embs = selected_model.encode(request.documents, show_progress_bar=False)
                return [
                    np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
                    for doc_emb in doc_embs
                ]

            scores = await run_sync(_rerank_embedding)

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

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Reranking error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reranking failed: {type(e).__name__}")
