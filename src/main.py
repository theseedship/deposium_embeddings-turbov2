"""
Deposium Embeddings Server - FastAPI application entry point.

All endpoint implementations live in src/routes/ modules.
Shared state (model_manager, auth) lives in src/shared.py.
Pydantic schemas live in src/schemas/requests.py.
"""
from fastapi import FastAPI
import torch
import logging
import asyncio

from . import shared
from .model_manager import get_model_manager
from .anthropic_compat import anthropic_router
from .anthropic_compat.router import set_dependencies as set_anthropic_dependencies
from .anthropic_compat.backends import BackendConfig, get_available_backends
from .routes import (
    catalog_router,
    embeddings_router,
    reranking_router,
    classification_router,
    vision_router,
    audio_router,
    benchmarks_router,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Deposium Embeddings - M2V-BGE-M3 + BGE-M3 ONNX + Anthropic API + Audio",
    description=(
        "M2V-BGE-M3-1024D (distilled from BGE-M3, ~3x more energy efficient) "
        "+ BGE-M3-ONNX INT8 (CPU high quality) + Qwen3 Reranking + VL Classifier "
        "+ Anthropic-compatible LLM API + Whisper Audio Transcription"
    ),
    version="13.0.0",
)

# Include all routers
app.include_router(anthropic_router)
app.include_router(catalog_router)
app.include_router(embeddings_router)
app.include_router(reranking_router)
app.include_router(classification_router)
app.include_router(vision_router)
app.include_router(audio_router)
app.include_router(benchmarks_router)


@app.on_event("startup")
async def initialize_models():
    logger.info("=" * 80)
    logger.info("Initializing Model Manager with Dynamic VRAM Management")
    logger.info("=" * 80)

    shared.model_manager = get_model_manager()

    # Initialize backend configuration from environment
    backend_config = BackendConfig.from_env()
    available_backends = get_available_backends()
    logger.info(f"LLM Backend: {backend_config.backend_type.value}")
    logger.info(f"Available backends: {[b.value for b in available_backends]}")

    # Set up Anthropic-compatible API dependencies
    set_anthropic_dependencies(shared.model_manager, shared.verify_api_key, backend_config)
    logger.info("Anthropic-compatible API initialized (/v1/messages)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if device == "cuda":
        used_mb, free_mb = shared.model_manager.get_vram_usage_mb()
        total_mb = used_mb + free_mb
        logger.info(f"GPU Memory: {used_mb}MB used, {free_mb}MB free (Total: {total_mb}MB)")
        logger.info(f"VRAM Limit: {shared.model_manager.max_vram_mb}MB (keeps 1GB margin)")

    logger.info("\nModel Loading Strategy:")
    logger.info("  - Lazy loading: Models loaded only when needed")
    logger.info("  - Priority system: High-priority models stay in VRAM")
    logger.info("  - Auto-unloading: Frees VRAM when limit exceeded")

    logger.info("\nAvailable Models:")
    logger.info("  - m2v-bge-m3-1024d: Distilled BGE-M3 embeddings (priority: 10)")
    logger.info("  - bge-m3-onnx: BGE-M3 ONNX INT8 for CPU (priority: 8)")
    logger.info("  - gemma-768d: Legacy multilingual embeddings (priority: 5)")
    logger.info("  - qwen3-rerank: Document reranking (priority: 8)")
    logger.info("  - vl-classifier: Document complexity classifier (ONNX, standalone)")

    # Start background cleanup task
    async def model_cleanup_loop():
        """Background task to cleanup inactive models."""
        logger.info("Starting model cleanup background task")
        while True:
            try:
                await asyncio.sleep(60)
                if shared.model_manager:
                    shared.model_manager.cleanup_inactive_models()
            except Exception as e:
                logger.error(f"Error in model cleanup loop: {e}")
                await asyncio.sleep(60)

    asyncio.create_task(model_cleanup_loop())
    logger.info("Model Manager initialized! Models will load on first use.")
