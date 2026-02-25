"""Embedding generation routes."""
import logging
import torch
from fastapi import APIRouter, Depends, HTTPException

from .. import shared
from ..shared import run_sync
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

        # Determine truncation: request.dimensions takes priority, then model config
        truncate_dims = request.dimensions or getattr(selected_model, '_truncate_dims', None)

        # Generate embeddings (CPU/GPU-heavy, offload to thread pool)
        def _encode():
            with torch.inference_mode():
                embs = selected_model.encode(texts, show_progress_bar=False)
            if truncate_dims:
                embs = embs[:, :truncate_dims]
            return [emb.tolist() for emb in embs]

        embeddings_list = await run_sync(_encode)

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

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Embedding generation failed")


@router.post("/api/embeddings")
async def create_embedding_alt(request: EmbedRequest, api_key: str = Depends(shared.verify_api_key)):
    """Alternative endpoint (some clients use /api/embeddings)"""
    return await create_embedding(request, api_key)
