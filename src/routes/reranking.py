"""Document reranking routes."""
import logging
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from .. import shared
from ..schemas.requests import RerankRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/rerank")
async def rerank_documents(request: RerankRequest, api_key: str = Depends(shared.verify_api_key)):
    """
    Rerank documents by relevance to a query.

    **Models:**
    - mxbai-rerank-v2: SOTA cross-encoder (BEIR 55.57, 100+ languages) - RECOMMENDED
    - mxbai-rerank-xsmall: Lighter cross-encoder (~40% faster, 278M params)
    - qwen3-rerank: Bi-encoder with cosine similarity (faster, lower quality)
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

        # MXBAI Reranker (true cross-encoder - SOTA quality)
        # Supports both V1 (DeBERTa) and V2 (Qwen2) architectures
        if model_config and model_config.type in ("mxbai_reranker", "mxbai_reranker_v1"):
            if not shared.MXBAI_RERANK_AVAILABLE:
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
