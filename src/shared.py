"""
Shared state and dependencies for all route modules.
"""
import asyncio
import functools
import os
import logging

from fastapi import HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# Rate limiter (shared across all routes)
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# Global model manager instance (set during startup)
model_manager = None

# Default model names from environment variables
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "m2v-bge-m3-1024d")
DEFAULT_RERANK_MODEL = os.getenv("DEFAULT_RERANK_MODEL", "bge-reranker-v2-m3")

# Whisper availability
try:
    from .audio import get_whisper_handler, WhisperHandler
    WHISPER_AVAILABLE = True
except ImportError:
    get_whisper_handler = None
    WhisperHandler = None
    WHISPER_AVAILABLE = False

# MXBAI Reranker availability
try:
    from mxbai_rerank import MxbaiRerankV2
    MXBAI_RERANK_AVAILABLE = True
except ImportError:
    MxbaiRerankV2 = None
    MXBAI_RERANK_AVAILABLE = False


async def verify_api_key(request: Request):
    """
    Verify API key from either Authorization Bearer or X-API-Key header.
    Railway internal network: bypassed. No EMBEDDINGS_API_KEY: dev mode.
    """
    host = request.headers.get("host", "")
    if ".railway.internal" in host:
        logger.info(f"Railway internal network request from {host} - authentication bypassed")
        return "railway-internal"

    expected_key = os.getenv("EMBEDDINGS_API_KEY")
    if not expected_key:
        logger.warning("EMBEDDINGS_API_KEY not configured - authentication disabled!")
        return "dev-mode"

    token = None
    auth_method = None

    authorization = request.headers.get("authorization")
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        auth_method = "Bearer"

    if not token:
        x_api_key = request.headers.get("x-api-key")
        if x_api_key:
            token = x_api_key
            auth_method = "X-API-Key"

    if not token or token != expected_key:
        logger.warning(f"Invalid API key attempt from {host} (method: {auth_method or 'none'})")
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Use 'Authorization: Bearer <token>' or 'X-API-Key: <token>' header",
            headers={"WWW-Authenticate": "Bearer"}
        )

    logger.info(f"Authentication successful from {host} (method: {auth_method})")
    return token


async def run_sync(func, *args, **kwargs):
    """Run a blocking function in a thread pool executor to avoid blocking the async event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, functools.partial(func, *args, **kwargs)
    )
