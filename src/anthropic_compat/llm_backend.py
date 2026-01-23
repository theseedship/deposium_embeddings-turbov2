"""
LLM Backend for Anthropic-compatible API.

Handles inference with HuggingFace causal language models,
supporting both synchronous generation and streaming.
"""

import logging
import time
from typing import Any, Dict, Generator, List, Optional, Tuple
import torch

logger = logging.getLogger(__name__)


class LLMBackend:
    """
    Backend for running inference on causal language models.

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
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

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
            max_length=self._get_max_context_length() - max_tokens
        ).to(self.device)

        input_tokens = inputs["input_ids"].shape[1]

        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 1e-7,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if top_k is not None:
            gen_kwargs["top_k"] = top_k

        # Add stop sequences if provided
        if stop_sequences:
            stop_ids = []
            for seq in stop_sequences:
                ids = self.tokenizer.encode(seq, add_special_tokens=False)
                if ids:
                    stop_ids.append(ids[0])
            if stop_ids:
                gen_kwargs["eos_token_id"] = [self.tokenizer.eos_token_id] + stop_ids

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

        return generated_text, input_tokens, output_tokens

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
        from threading import Thread
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
            max_length=self._get_max_context_length() - max_tokens
        ).to(self.device)

        input_tokens = inputs["input_ids"].shape[1]

        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 1e-7,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

        if top_k is not None:
            gen_kwargs["top_k"] = top_k

        # Add stop sequences if provided
        if stop_sequences:
            stop_ids = []
            for seq in stop_sequences:
                ids = self.tokenizer.encode(seq, add_special_tokens=False)
                if ids:
                    stop_ids.append(ids[0])
            if stop_ids:
                gen_kwargs["eos_token_id"] = [self.tokenizer.eos_token_id] + stop_ids

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
            if stop_sequences:
                for seq in stop_sequences:
                    if seq in accumulated_text:
                        # Truncate at stop sequence
                        idx = accumulated_text.find(seq)
                        text = accumulated_text[:idx]
                        should_stop = True
                        break

            yield text, False, input_tokens, output_tokens

            if should_stop:
                break

        thread.join()

        # Final yield
        yield "", True, input_tokens, output_tokens

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

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
