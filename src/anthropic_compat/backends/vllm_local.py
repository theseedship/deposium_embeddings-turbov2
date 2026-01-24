"""
vLLM local inference backend.

Uses vLLM for high-performance inference with:
- PagedAttention for efficient KV-cache management
- Continuous batching for high throughput
- Support for AWQ, GPTQ, bitsandbytes quantization
- Multi-GPU tensor parallelism
"""

import asyncio
import logging
from typing import Any, Dict, Generator, List, Optional

from .base import (
    BackendCapabilities,
    GenerationConfig,
    GenerationResult,
    InferenceBackend,
    StreamChunk,
)
from .config import VLLMLocalConfig

logger = logging.getLogger(__name__)

# vLLM is an optional dependency
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    AsyncLLMEngine = None
    AsyncEngineArgs = None


class VLLMLocalBackend(InferenceBackend):
    """
    vLLM local inference backend.

    Features:
    - PagedAttention: <4% KV-cache waste vs 60-80% in HuggingFace
    - Continuous batching: 24x throughput improvement
    - AWQ/GPTQ/bitsandbytes quantization
    - Multi-GPU tensor parallelism
    - Prefix caching for repeated prompts
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name: Optional[str] = None,
        config: Optional[VLLMLocalConfig] = None,
    ):
        """
        Initialize the vLLM backend.

        Args:
            model_name_or_path: HuggingFace model ID or local path
            tokenizer_name: Optional tokenizer name (defaults to model)
            config: vLLM configuration
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm>=0.6.0\n"
                "Note: vLLM requires CUDA 12.1+ and 8GB+ VRAM."
            )

        self.model_name = model_name_or_path
        self.tokenizer_name = tokenizer_name or model_name_or_path
        self.config = config or VLLMLocalConfig.from_env()

        # Initialize vLLM engine
        self._engine: Optional[LLM] = None
        self._async_engine: Optional[AsyncLLMEngine] = None
        self._tokenizer = None
        self._capabilities: Optional[BackendCapabilities] = None

        # Lazy initialization
        self._initialized = False

    def _ensure_initialized(self):
        """Lazily initialize the vLLM engine."""
        if self._initialized:
            return

        logger.info(f"Initializing vLLM engine for {self.model_name}")

        # Build engine kwargs
        engine_kwargs = {
            "model": self.model_name,
            "tokenizer": self.tokenizer_name,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "dtype": self.config.dtype,
            "trust_remote_code": True,
            "enforce_eager": self.config.enforce_eager,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "max_num_seqs": self.config.max_num_seqs,
        }

        if self.config.max_model_len:
            engine_kwargs["max_model_len"] = self.config.max_model_len

        if self.config.quantization:
            engine_kwargs["quantization"] = self.config.quantization

        try:
            self._engine = LLM(**engine_kwargs)
            self._tokenizer = self._engine.get_tokenizer()
            self._initialized = True
            self._capabilities = self._build_capabilities()
            logger.info(f"vLLM engine initialized successfully for {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise

    def _build_capabilities(self) -> BackendCapabilities:
        """Build capabilities based on vLLM config."""
        quantization_methods = ["awq", "gptq", "bitsandbytes", "marlin", "fp8"]

        # Get max context from model config
        max_ctx = 4096
        if self._engine:
            model_config = self._engine.llm_engine.model_config
            max_ctx = getattr(model_config, "max_model_len", 4096)

        return BackendCapabilities(
            streaming=True,
            tool_calling=True,
            batching=True,
            multi_gpu=self.config.tensor_parallel_size > 1,
            quantization=quantization_methods,
            max_context_length=max_ctx,
            continuous_batching=True,
            paged_attention=True,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return the capabilities of this backend."""
        if self._capabilities is None:
            self._ensure_initialized()
        return self._capabilities

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        self._ensure_initialized()
        return len(self._tokenizer.encode(text))

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
        import time

        self._ensure_initialized()

        start_time = time.time()

        # Apply chat template
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        input_tokens = len(self._tokenizer.encode(prompt))

        # Build sampling params
        sampling_params = self._build_sampling_params(config)

        # Generate
        outputs = self._engine.generate(prompt, sampling_params)
        output = outputs[0]

        generated_text = output.outputs[0].text
        output_tokens = len(output.outputs[0].token_ids)

        elapsed_ms = (time.time() - start_time) * 1000
        tokens_per_second = output_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        # Determine finish reason
        finish_reason = self._map_finish_reason(output.outputs[0].finish_reason)

        return GenerationResult(
            text=generated_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            latency_ms=elapsed_ms,
            tokens_per_second=tokens_per_second,
        )

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
    ) -> Generator[StreamChunk, None, None]:
        """
        Generate a response with streaming.

        Uses synchronous streaming for compatibility with existing code.
        For async streaming, use generate_stream_async.

        Args:
            messages: Chat messages in HuggingFace format
            config: Generation configuration

        Yields:
            StreamChunk objects with text chunks and metadata
        """
        self._ensure_initialized()

        # Apply chat template
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        input_tokens = len(self._tokenizer.encode(prompt))

        # Build sampling params
        sampling_params = self._build_sampling_params(config)

        # Use vLLM's built-in streaming (requires engine in streaming mode)
        # For simpler implementation, we do full generation then yield
        # A production implementation would use AsyncLLMEngine
        outputs = self._engine.generate(prompt, sampling_params)
        output = outputs[0]

        generated_text = output.outputs[0].text
        output_tokens = len(output.outputs[0].token_ids)

        # Simulate streaming by yielding chunks
        # This is a simplified implementation - for true streaming,
        # use the async engine or vLLM's streaming API
        chunk_size = 4
        for i in range(0, len(generated_text), chunk_size):
            chunk = generated_text[i:i + chunk_size]
            is_final = i + chunk_size >= len(generated_text)

            yield StreamChunk(
                text=chunk,
                is_final=False,
                input_tokens=input_tokens,
                output_tokens=min(output_tokens, (i + chunk_size) // 4),
            )

        # Final chunk
        finish_reason = self._map_finish_reason(output.outputs[0].finish_reason)

        yield StreamChunk(
            text="",
            is_final=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )

    async def generate_stream_async(
        self,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
    ):
        """
        Generate a response with true async streaming.

        This uses vLLM's AsyncLLMEngine for proper streaming.

        Args:
            messages: Chat messages in HuggingFace format
            config: Generation configuration

        Yields:
            StreamChunk objects with text chunks and metadata
        """
        if self._async_engine is None:
            # Initialize async engine
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                tokenizer=self.tokenizer_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                dtype=self.config.dtype,
                trust_remote_code=True,
                max_num_seqs=self.config.max_num_seqs,
            )
            if self.config.quantization:
                engine_args.quantization = self.config.quantization

            self._async_engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Apply chat template
        tokenizer = await self._async_engine.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        input_tokens = len(tokenizer.encode(prompt))

        # Build sampling params
        sampling_params = self._build_sampling_params(config)

        # Generate with streaming
        import uuid
        request_id = str(uuid.uuid4())

        output_tokens = 0
        async for output in self._async_engine.generate(prompt, sampling_params, request_id):
            if output.outputs:
                text = output.outputs[0].text
                output_tokens = len(output.outputs[0].token_ids)
                finished = output.finished

                yield StreamChunk(
                    text=text,
                    is_final=finished,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    finish_reason=self._map_finish_reason(output.outputs[0].finish_reason) if finished else None,
                )

    def _build_sampling_params(self, config: GenerationConfig) -> "SamplingParams":
        """Build vLLM sampling parameters from config."""
        kwargs = {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature if config.temperature > 0 else 0.0,
            "top_p": config.top_p,
        }

        if config.top_k is not None:
            kwargs["top_k"] = config.top_k

        if config.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = config.repetition_penalty

        if config.presence_penalty != 0.0:
            kwargs["presence_penalty"] = config.presence_penalty

        if config.frequency_penalty != 0.0:
            kwargs["frequency_penalty"] = config.frequency_penalty

        if config.stop_sequences:
            kwargs["stop"] = config.stop_sequences

        return SamplingParams(**kwargs)

    def _map_finish_reason(self, vllm_reason: Optional[str]) -> str:
        """Map vLLM finish reason to our standard format."""
        if vllm_reason is None:
            return "stop"

        mapping = {
            "stop": "stop",
            "length": "length",
            "abort": "stop",
        }
        return mapping.get(str(vllm_reason).lower(), "stop")

    def is_available(self) -> bool:
        """Check if vLLM is available."""
        return VLLM_AVAILABLE

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._engine:
            # vLLM doesn't have explicit cleanup, but we can clear references
            self._engine = None
            self._async_engine = None
            self._tokenizer = None
            self._initialized = False

            # Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        logger.info("vLLM engine shutdown complete")
