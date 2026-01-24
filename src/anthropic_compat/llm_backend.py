"""
LLM Backend for Anthropic-compatible API.

DEPRECATED: This module is maintained for backward compatibility.
New code should use the backends module directly:

    from anthropic_compat.backends import (
        create_backend,
        HuggingFaceBackend,
        GenerationConfig,
    )

Handles inference with HuggingFace causal language models,
supporting both synchronous generation and streaming.
"""

import warnings
from typing import Any, Dict, Generator, List, Optional, Tuple

# Import from new backends module for compatibility
from .backends import HuggingFaceBackend, GenerationConfig

__all__ = ["LLMBackend"]


class LLMBackend:
    """
    Backward-compatible wrapper around HuggingFaceBackend.

    DEPRECATED: Use HuggingFaceBackend or create_backend() instead.

    Supports:
    - Qwen2.5-Coder and similar instruction-tuned models
    - 4-bit quantization via bitsandbytes
    - Streaming generation with TextIteratorStreamer
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        """
        Initialize the LLM backend.

        Args:
            model: HuggingFace model (AutoModelForCausalLM)
            tokenizer: HuggingFace tokenizer
            device: Device to run inference on
        """
        warnings.warn(
            "LLMBackend is deprecated. Use HuggingFaceBackend or create_backend() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._backend = HuggingFaceBackend(model, tokenizer, device)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> Tuple[str, int, int]:
        """
        Generate a response synchronously.

        Args:
            messages: Chat messages in HuggingFace format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: Sequences that stop generation

        Returns:
            Tuple of (generated_text, input_tokens, output_tokens)
        """
        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
        )

        result = self._backend.generate(messages, config)
        return result.text, result.input_tokens, result.output_tokens

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> Generator[Tuple[str, bool, int, int], None, None]:
        """
        Generate a response with streaming.

        Args:
            messages: Chat messages in HuggingFace format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: Sequences that stop generation

        Yields:
            Tuples of (text_chunk, is_final, input_tokens, output_tokens)
        """
        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
        )

        for chunk in self._backend.generate_stream(messages, config):
            yield chunk.text, chunk.is_final, chunk.input_tokens, chunk.output_tokens

    def _get_max_context_length(self) -> int:
        """Get the model's maximum context length."""
        return self._backend.capabilities.max_context_length

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return self._backend.count_tokens(text)
