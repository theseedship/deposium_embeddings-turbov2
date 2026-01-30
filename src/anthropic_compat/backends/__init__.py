"""
Inference backend abstraction layer.

Provides a unified interface for multiple inference engines:
- HuggingFace Transformers (default)
- vLLM Local (high-performance)
- Remote OpenAI-compatible APIs (vLLM server, SGLang, cloud)

Usage:
    from anthropic_compat.backends import create_backend, BackendType

    # Auto-detect best backend
    backend = create_backend(model, tokenizer)

    # Specify backend type
    backend = create_backend(
        model, tokenizer,
        backend_type=BackendType.VLLM_LOCAL
    )

    # Generate
    result = backend.generate(messages, GenerationConfig())
"""

import logging
import os
from typing import Any, Optional, Union

from .base import (
    BackendCapabilities,
    GenerationConfig,
    GenerationResult,
    InferenceBackend,
    StreamChunk,
)
from .config import (
    BackendConfig,
    BackendType,
    BitNetConfig,
    HuggingFaceConfig,
    QuantizationType,
    RemoteOpenAIConfig,
    VLLMLocalConfig,
    VLLMRemoteConfig,
)
from .huggingface import HuggingFaceBackend

logger = logging.getLogger(__name__)

# Lazy imports for optional backends
_vllm_backend_class = None
_remote_backend_class = None
_bitnet_backend_class = None


def _get_vllm_backend_class():
    """Lazy import vLLM backend."""
    global _vllm_backend_class
    if _vllm_backend_class is None:
        from .vllm_local import VLLMLocalBackend

        _vllm_backend_class = VLLMLocalBackend
    return _vllm_backend_class


def _get_remote_backend_class():
    """Lazy import remote backend."""
    global _remote_backend_class
    if _remote_backend_class is None:
        from .remote_openai import RemoteOpenAIBackend

        _remote_backend_class = RemoteOpenAIBackend
    return _remote_backend_class


def _get_bitnet_backend_class():
    """Lazy import BitNet backend."""
    global _bitnet_backend_class
    if _bitnet_backend_class is None:
        from .bitnet import BitNetBackend

        _bitnet_backend_class = BitNetBackend
    return _bitnet_backend_class


def create_backend(
    model_or_path: Any,
    tokenizer: Optional[Any] = None,
    backend_type: Optional[BackendType] = None,
    config: Optional[BackendConfig] = None,
    device: str = "cuda",
) -> InferenceBackend:
    """
    Factory function to create an inference backend.

    This is the main entry point for creating backends. It handles:
    - Auto-detection of best available backend
    - Configuration from environment variables
    - Graceful fallback if preferred backend unavailable

    Args:
        model_or_path: HuggingFace model object or model name/path for vLLM
        tokenizer: HuggingFace tokenizer (required for HuggingFace backend)
        backend_type: Explicitly specify backend type (auto-detect if None)
        config: Backend configuration (loads from env if None)
        device: Device for inference (HuggingFace backend only)

    Returns:
        InferenceBackend instance

    Examples:
        # HuggingFace backend with loaded model
        backend = create_backend(model, tokenizer)

        # vLLM local with model path
        backend = create_backend(
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            backend_type=BackendType.VLLM_LOCAL
        )

        # Remote vLLM server
        backend = create_backend(
            "qwen2.5-coder-7b",
            backend_type=BackendType.VLLM_REMOTE
        )
    """
    # Load config from environment if not provided
    if config is None:
        config = BackendConfig.from_env()

    # Determine backend type
    if backend_type is None:
        backend_type = config.backend_type

    logger.info(f"Creating {backend_type.value} backend")

    try:
        if backend_type == BackendType.HUGGINGFACE:
            return _create_huggingface_backend(
                model_or_path, tokenizer, config.huggingface, device
            )

        elif backend_type == BackendType.VLLM_LOCAL:
            return _create_vllm_local_backend(model_or_path, config.vllm_local)

        elif backend_type == BackendType.VLLM_REMOTE:
            return _create_remote_backend(
                model_or_path, config.vllm_remote
            )

        elif backend_type == BackendType.REMOTE_OPENAI:
            return _create_remote_backend(
                model_or_path, config.remote_openai
            )

        elif backend_type == BackendType.BITNET:
            return _create_bitnet_backend(model_or_path, config.bitnet)

        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    except Exception as e:
        logger.error(f"Failed to create {backend_type.value} backend: {e}")

        # Fallback to HuggingFace if we have model and tokenizer
        if backend_type != BackendType.HUGGINGFACE and tokenizer is not None:
            logger.warning("Falling back to HuggingFace backend")
            return _create_huggingface_backend(
                model_or_path, tokenizer, config.huggingface, device
            )

        raise


def _create_huggingface_backend(
    model: Any,
    tokenizer: Any,
    config: HuggingFaceConfig,
    device: str,
) -> HuggingFaceBackend:
    """Create HuggingFace backend."""
    if tokenizer is None:
        raise ValueError("HuggingFace backend requires a tokenizer")

    return HuggingFaceBackend(
        model=model,
        tokenizer=tokenizer,
        device=device,
        config=config,
    )


def _create_vllm_local_backend(
    model_path: str,
    config: VLLMLocalConfig,
) -> InferenceBackend:
    """Create vLLM local backend."""
    VLLMLocalBackend = _get_vllm_backend_class()

    # model_path should be a string (HuggingFace ID or local path)
    if not isinstance(model_path, str):
        # If it's a loaded model, try to get the name from config
        if hasattr(model_path, "config") and hasattr(model_path.config, "_name_or_path"):
            model_path = model_path.config._name_or_path
        else:
            raise ValueError(
                "vLLM backend requires a model path string, not a loaded model. "
                "Pass the HuggingFace model ID or local path."
            )

    return VLLMLocalBackend(
        model_name_or_path=model_path,
        config=config,
    )


def _create_remote_backend(
    model_name: str,
    config: Union[VLLMRemoteConfig, RemoteOpenAIConfig],
) -> InferenceBackend:
    """Create remote API backend."""
    RemoteOpenAIBackend = _get_remote_backend_class()

    # model_name should be a string
    if not isinstance(model_name, str):
        if hasattr(model_name, "config") and hasattr(model_name.config, "_name_or_path"):
            model_name = model_name.config._name_or_path
        else:
            model_name = str(model_name)

    return RemoteOpenAIBackend(
        model_name=model_name,
        config=config,
    )


def _create_bitnet_backend(
    model_path: str,
    config: "BitNetConfig",
) -> InferenceBackend:
    """Create BitNet CPU backend."""
    BitNetBackend = _get_bitnet_backend_class()

    # model_path should be a string path to GGUF file
    if not isinstance(model_path, str):
        # If not a string, use config's model_path
        model_path = config.model_path

    return BitNetBackend(
        model_path=model_path,
        config=config,
    )


def is_backend_available(backend_type: BackendType) -> bool:
    """
    Check if a backend type is available.

    Args:
        backend_type: The backend type to check

    Returns:
        True if the backend can be used
    """
    if backend_type == BackendType.HUGGINGFACE:
        return True  # Always available

    elif backend_type == BackendType.VLLM_LOCAL:
        try:
            from vllm import LLM

            return True
        except ImportError:
            return False

    elif backend_type in (BackendType.VLLM_REMOTE, BackendType.REMOTE_OPENAI):
        # Check if httpx is available (it should be via FastAPI)
        try:
            import httpx

            return True
        except ImportError:
            return False

    elif backend_type == BackendType.BITNET:
        # Check if BitNet paths are configured and exist
        from pathlib import Path
        config = BitNetConfig.from_env()
        bitnet_path = Path(config.bitnet_path)
        model_path = Path(config.model_path)
        return bitnet_path.exists() and model_path.exists()

    return False


def get_available_backends() -> list[BackendType]:
    """
    Get list of available backend types.

    Returns:
        List of BackendType that can be used
    """
    return [bt for bt in BackendType if is_backend_available(bt)]


# Public exports
__all__ = [
    # Factory
    "create_backend",
    "is_backend_available",
    "get_available_backends",
    # Base classes
    "InferenceBackend",
    "BackendCapabilities",
    "GenerationConfig",
    "GenerationResult",
    "StreamChunk",
    # Config
    "BackendConfig",
    "BackendType",
    "BitNetConfig",
    "HuggingFaceConfig",
    "VLLMLocalConfig",
    "VLLMRemoteConfig",
    "RemoteOpenAIConfig",
    "QuantizationType",
    # Backend implementations
    "HuggingFaceBackend",
]
