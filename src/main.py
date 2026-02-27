"""
Deposium Embeddings Server - FastAPI application entry point.

All endpoint implementations live in src/routes/ modules.
Shared state (model_manager, auth) lives in src/shared.py.
Pydantic schemas live in src/schemas/requests.py.
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
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

# --- CORS Middleware ---
# Configurable via CORS_ALLOWED_ORIGINS env var (comma-separated)
# Default: restrict to same-origin only (empty list = no cross-origin)
_cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _cors_origins.split(",") if o.strip()] if _cors_origins else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type"],
)

# --- Rate Limiting Middleware (slowapi) ---
# Default: 200 requests/minute per IP (configurable per-route via @limiter.limit)
app.state.limiter = shared.limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

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
    logger.info("  - m2v-bge-m3-1024d: Distilled BGE-M3 embeddings")
    logger.info("  - bge-m3-onnx: BGE-M3 ONNX INT8 for CPU")
    logger.info("  - bge-m3-matryoshka: BGE-M3 Matryoshka ONNX INT8 (FR fine-tuned)")
    logger.info("  - bge-reranker-v2-m3: BGE-Reranker-v2-m3 ONNX INT8 (DEFAULT reranker)")
    logger.info("  - vl-classifier: Document complexity classifier (ONNX, standalone)")

    # Preload critical models at startup (eliminates cold-start latency)
    preload_models = os.getenv("PRELOAD_MODELS", "bge-m3-matryoshka,bge-reranker-v2-m3")
    if preload_models:
        for model_name in preload_models.split(","):
            model_name = model_name.strip()
            if model_name and model_name in shared.model_manager.configs:
                try:
                    t0 = asyncio.get_event_loop().time()
                    shared.model_manager.get_model(model_name)
                    dt = (asyncio.get_event_loop().time() - t0) * 1000
                    logger.info(f"  ✅ Preloaded {model_name} ({dt:.0f}ms)")
                except Exception as e:
                    logger.warning(f"  ⚠️  Could not preload {model_name}: {e}")
            elif model_name:
                logger.warning(f"  ⚠️  Unknown model to preload: {model_name}")

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

    # Store task reference to prevent garbage collection (asyncio holds only a weak ref)
    app.state.cleanup_task = asyncio.create_task(model_cleanup_loop())
    logger.info("Model Manager initialized! Models will load on first use.")


@app.on_event("shutdown")
async def graceful_shutdown():
    """Clean up resources on SIGTERM/SIGINT."""
    logger.info("Graceful shutdown initiated...")

    # Cancel background cleanup task
    if hasattr(app.state, "cleanup_task") and app.state.cleanup_task:
        app.state.cleanup_task.cancel()
        try:
            await app.state.cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Background cleanup task cancelled")

    # Unload all models
    if shared.model_manager:
        for name in list(shared.model_manager.models.keys()):
            try:
                shared.model_manager._unload_model(name)
            except Exception as e:
                logger.warning(f"Error unloading model {name}: {e}")
        logger.info("All models unloaded")

    # Flush CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")

    logger.info("Graceful shutdown complete")
