"""
HuggingFace Transformers inference backend.

Uses the standard HuggingFace transformers library for inference.
Supports bitsandbytes quantization (NF4/INT8) for memory efficiency.
"""

import logging
import time
from threading import Thread
from typing import Any, Dict, Generator, List, Optional

import torch

from .base import (
    BackendCapabilities,
    GenerationConfig,
    GenerationResult,
    InferenceBackend,
    StreamChunk,
)
from .config import HuggingFaceConfig

logger = logging.getLogger(__name__)


class HuggingFaceBackend(InferenceBackend):
    """
    HuggingFace Transformers backend.

    Features:
    - Standard transformers library inference
    - bitsandbytes NF4/INT8 quantization support
    - TextIteratorStreamer for streaming
    - Flash Attention 2 support
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        config: Optional[HuggingFaceConfig] = None,
    ):
        """
        Initialize the HuggingFace backend.

        Args:
            model: HuggingFace model (AutoModelForCausalLM)
            tokenizer: HuggingFace tokenizer
            device: Device to run inference on
            config: Backend configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or HuggingFaceConfig()
        self._capabilities = self._build_capabilities()

    def _build_capabilities(self) -> BackendCapabilities:
        """Build capabilities based on model config."""
        max_ctx = self._get_max_context_length()

        quantization_methods = ["bitsandbytes"]
        if hasattr(self.model, "quantization_config"):
            q_config = getattr(self.model, "quantization_config", None)
            if q_config and hasattr(q_config, "quant_method"):
                quantization_methods.append(q_config.quant_method)

        return BackendCapabilities(
            streaming=True,
            tool_calling=True,
            batching=False,  # HF generate() is single-batch by default
            multi_gpu=False,  # Would need device_map for multi-GPU
            quantization=quantization_methods,
            max_context_length=max_ctx,
            continuous_batching=False,
            paged_attention=False,
        )

    def _get_max_context_length(self) -> int:
        """Get the model's maximum context length."""
        if hasattr(self.model.config, "max_position_embeddings"):
            return self.model.config.max_position_embeddings
        elif hasattr(self.model.config, "max_length"):
            return self.model.config.max_length
        elif hasattr(self.model.config, "n_positions"):
            return self.model.config.n_positions
        else:
            return 4096  # Conservative default

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return the capabilities of this backend."""
        return self._capabilities

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

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
        start_time = time.time()

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._get_max_context_length() - config.max_tokens
        ).to(self.device)

        input_tokens = inputs["input_ids"].shape[1]

        # Build generation kwargs
        gen_kwargs = self._build_gen_kwargs(config)

        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )

        # Decode only the new tokens
        generated_ids = outputs[0][input_tokens:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        output_tokens = len(generated_ids)

        elapsed_ms = (time.time() - start_time) * 1000
        tokens_per_second = output_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        # Determine finish reason
        finish_reason = "stop"
        if output_tokens >= config.max_tokens:
            finish_reason = "length"

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

        Args:
            messages: Chat messages in HuggingFace format
            config: Generation configuration

        Yields:
            StreamChunk objects with text chunks and metadata
        """
        from transformers import TextIteratorStreamer

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._get_max_context_length() - config.max_tokens
        ).to(self.device)

        input_tokens = inputs["input_ids"].shape[1]

        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Build generation kwargs
        gen_kwargs = self._build_gen_kwargs(config)
        gen_kwargs["streamer"] = streamer

        # Start generation in a separate thread
        def generate_async():
            with torch.inference_mode():
                self.model.generate(**inputs, **gen_kwargs)

        thread = Thread(target=generate_async)
        thread.start()

        # Stream tokens
        output_tokens = 0
        accumulated_text = ""

        for text in streamer:
            accumulated_text += text
            output_tokens += 1  # Approximate - streamer doesn't give exact count

            # Check for stop sequences in accumulated text
            should_stop = False
            if config.stop_sequences:
                for seq in config.stop_sequences:
                    if seq in accumulated_text:
                        # Truncate at stop sequence
                        idx = accumulated_text.find(seq)
                        text = accumulated_text[:idx]
                        should_stop = True
                        break

            yield StreamChunk(
                text=text,
                is_final=False,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            if should_stop:
                break

        thread.join()

        # Final chunk
        finish_reason = "stop"
        if output_tokens >= config.max_tokens:
            finish_reason = "length"

        yield StreamChunk(
            text="",
            is_final=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )

    def _build_gen_kwargs(self, config: GenerationConfig) -> Dict[str, Any]:
        """Build generation kwargs from config."""
        gen_kwargs = {
            "max_new_tokens": config.max_tokens,
            "temperature": config.temperature if config.temperature > 0 else 1e-7,
            "top_p": config.top_p,
            "do_sample": config.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if config.top_k is not None:
            gen_kwargs["top_k"] = config.top_k

        if config.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = config.repetition_penalty

        # Add stop sequences if provided
        if config.stop_sequences:
            stop_ids = []
            for seq in config.stop_sequences:
                ids = self.tokenizer.encode(seq, add_special_tokens=False)
                if ids:
                    stop_ids.append(ids[0])
            if stop_ids:
                gen_kwargs["eos_token_id"] = [self.tokenizer.eos_token_id] + stop_ids

        return gen_kwargs

    def shutdown(self) -> None:
        """Clean up resources."""
        # Clear CUDA cache if using GPU
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
