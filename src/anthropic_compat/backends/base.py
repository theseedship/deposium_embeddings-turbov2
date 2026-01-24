"""
Abstract base class for inference backends.

Defines the interface that all inference backends must implement,
enabling seamless switching between HuggingFace, vLLM, and remote APIs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple


@dataclass
class BackendCapabilities:
    """Describes what a backend can do."""

    streaming: bool = True
    tool_calling: bool = True
    batching: bool = False
    multi_gpu: bool = False
    quantization: List[str] = field(default_factory=list)
    max_context_length: int = 4096

    # Performance characteristics
    continuous_batching: bool = False
    paged_attention: bool = False


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None

    # Additional parameters for advanced backends
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


@dataclass
class GenerationResult:
    """Result from a generation call."""

    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str = "stop"  # "stop", "length", "tool_calls"

    # Optional metadata
    latency_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None


@dataclass
class StreamChunk:
    """A chunk from streaming generation."""

    text: str
    is_final: bool
    input_tokens: int
    output_tokens: int
    finish_reason: Optional[str] = None


class InferenceBackend(ABC):
    """
    Abstract base class for inference backends.

    All backends must implement:
    - generate(): Synchronous generation
    - generate_stream(): Streaming generation
    - capabilities: Property describing backend features
    """

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
    ) -> GenerationResult:
        """
        Generate a response synchronously.

        Args:
            messages: Chat messages in HuggingFace format
            config: Generation configuration

        Returns:
            GenerationResult with generated text and metadata
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
    ) -> Generator[StreamChunk, None, None]:
        """
        Generate a response with streaming.

        Args:
            messages: Chat messages in HuggingFace format
            config: Generation configuration

        Yields:
            StreamChunk objects with text chunks and metadata
        """
        pass

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return the capabilities of this backend."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        pass

    def get_max_context_length(self) -> int:
        """Get the maximum context length supported."""
        return self.capabilities.max_context_length

    def is_available(self) -> bool:
        """Check if the backend is available and ready."""
        return True

    def shutdown(self) -> None:
        """Clean up resources. Override in subclasses if needed."""
        pass
