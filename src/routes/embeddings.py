"""Embedding generation routes."""
import logging
from fastapi import APIRouter, Depends, HTTPException

from .. import shared
from ..schemas.requests import EmbedRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/embed")
async def create_embedding(request: EmbedRequest, api_key: str = Depends(shared.verify_api_key)):
    """Ollama-compatible embedding endpoint with multi-model support"""

    # Validate model selection
    available_models = shared.model_manager.configs.keys()
    if request.model not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found. Available: {list(available_models)}"
        )

    try:
        # Handle both string and list inputs
        texts = [request.input] if isinstance(request.input, str) else request.input

        # Get model (lazy loading)
        selected_model = shared.model_manager.get_model(request.model)

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

        # Return both OpenAI format (embeddings) and Ollama format (embedding)
        # This ensures compatibility with clients expecting either format
        response = {
            "model": request.model,
            "embeddings": embeddings_list,  # OpenAI format: array of arrays
        }
        # Add Ollama format: single array (first embedding) for single input
        if len(embeddings_list) == 1:
            response["embedding"] = embeddings_list[0]

        return response

    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/embeddings")
async def create_embedding_alt(request: EmbedRequest, api_key: str = Depends(shared.verify_api_key)):
    """Alternative endpoint (some clients use /api/embeddings)"""
    return await create_embedding(request, api_key)
