"""
Remote OpenAI-compatible API backend.

Supports any endpoint that implements the OpenAI Chat Completions API:
- vLLM server (/v1/chat/completions)
- SGLang server
- Ollama (with OpenAI compatibility mode)
- Cloud providers (Together, Anyscale, Fireworks, etc.)
- OpenAI itself
"""

import asyncio
import logging
import time
from typing import Any, Dict, Generator, List, Optional, Union

import httpx

from .base import (
    BackendCapabilities,
    GenerationConfig,
    GenerationResult,
    InferenceBackend,
    StreamChunk,
)
from .config import RemoteOpenAIConfig, VLLMRemoteConfig

logger = logging.getLogger(__name__)

# Optional dependency
try:
    from openai import OpenAI, AsyncOpenAI
    from openai import APIError, APIConnectionError, RateLimitError

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    AsyncOpenAI = None


class RemoteOpenAIBackend(InferenceBackend):
    """
    Remote OpenAI-compatible API backend.

    Supports:
    - vLLM served via OpenAI-compatible API
    - SGLang, TGI with OpenAI wrapper
    - Cloud endpoints (Together, Anyscale, etc.)
    - Rate limiting and retry logic
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[Union[RemoteOpenAIConfig, VLLMRemoteConfig]] = None,
    ):
        """
        Initialize the remote backend.

        Args:
            model_name: Model name to use in API requests
            config: Remote API configuration
        """
        self.model_name = model_name
        self.config = config or RemoteOpenAIConfig.from_env()

        # Determine base URL and API key
        if isinstance(self.config, VLLMRemoteConfig):
            self.base_url = self.config.base_url
            self.api_key = self.config.api_key or "EMPTY"
            self.timeout = self.config.timeout
            self.max_retries = self.config.max_retries
            self.retry_delay = self.config.retry_delay
        else:
            self.base_url = self.config.base_url
            self.api_key = self.config.api_key or "EMPTY"
            self.timeout = self.config.timeout
            self.max_retries = self.config.max_retries
            self.retry_delay = 1.0
            if self.config.model_override:
                self.model_name = self.config.model_override

        # Initialize clients
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._openai_client: Optional["OpenAI"] = None
        self._openai_async_client: Optional["AsyncOpenAI"] = None

        self._capabilities = BackendCapabilities(
            streaming=True,
            tool_calling=True,
            batching=True,  # Server handles batching
            multi_gpu=True,  # Server may use multiple GPUs
            quantization=["awq", "gptq", "fp8"],  # Depends on server config
            max_context_length=32768,  # Assume large context, server will reject if too long
            continuous_batching=True,
            paged_attention=True,
        )

    def _get_sync_client(self) -> httpx.Client:
        """Get or create sync HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        return self._async_client

    def _get_openai_client(self) -> "OpenAI":
        """Get or create OpenAI SDK client."""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI SDK is not installed. Install with: pip install openai>=1.50.0"
            )

        if self._openai_client is None:
            self._openai_client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._openai_client

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return the capabilities of this backend."""
        return self._capabilities

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count.

        Note: Without access to the actual tokenizer, we use a rough estimate.
        For accurate counts, use the /tokenize endpoint if available.
        """
        # Rough estimate: 4 characters per token on average
        return len(text) // 4

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

        # Convert messages to OpenAI format (same as HF for chat)
        openai_messages = self._convert_messages(messages)

        # Try with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                if OPENAI_AVAILABLE:
                    result = self._generate_with_openai(openai_messages, config)
                else:
                    result = self._generate_with_httpx(openai_messages, config)

                elapsed_ms = (time.time() - start_time) * 1000
                result.latency_ms = elapsed_ms
                if result.output_tokens > 0:
                    result.tokens_per_second = result.output_tokens / (elapsed_ms / 1000)

                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"All {self.max_retries} attempts failed. Last error: {last_error}")

    def _generate_with_openai(
        self,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
    ) -> GenerationResult:
        """Generate using OpenAI SDK."""
        client = self._get_openai_client()

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        if config.stop_sequences:
            kwargs["stop"] = config.stop_sequences

        if config.presence_penalty != 0.0:
            kwargs["presence_penalty"] = config.presence_penalty

        if config.frequency_penalty != 0.0:
            kwargs["frequency_penalty"] = config.frequency_penalty

        response = client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        usage = response.usage

        return GenerationResult(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            finish_reason=self._map_finish_reason(choice.finish_reason),
        )

    def _generate_with_httpx(
        self,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
    ) -> GenerationResult:
        """Generate using raw HTTP requests (fallback without OpenAI SDK)."""
        client = self._get_sync_client()

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": False,
        }

        if config.stop_sequences:
            payload["stop"] = config.stop_sequences

        response = client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return GenerationResult(
            text=choice["message"]["content"],
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=self._map_finish_reason(choice.get("finish_reason")),
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
        openai_messages = self._convert_messages(messages)

        if OPENAI_AVAILABLE:
            yield from self._stream_with_openai(openai_messages, config)
        else:
            yield from self._stream_with_httpx(openai_messages, config)

    def _stream_with_openai(
        self,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
    ) -> Generator[StreamChunk, None, None]:
        """Stream using OpenAI SDK."""
        client = self._get_openai_client()

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if config.stop_sequences:
            kwargs["stop"] = config.stop_sequences

        stream = client.chat.completions.create(**kwargs)

        output_tokens = 0
        input_tokens = 0
        finish_reason = None

        for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    output_tokens += 1  # Approximate
                    yield StreamChunk(
                        text=delta.content,
                        is_final=False,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            # Handle usage in final chunk
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

        # Final chunk
        yield StreamChunk(
            text="",
            is_final=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=self._map_finish_reason(finish_reason),
        )

    def _stream_with_httpx(
        self,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
    ) -> Generator[StreamChunk, None, None]:
        """Stream using raw HTTP requests."""
        import json

        client = self._get_sync_client()

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
        }

        if config.stop_sequences:
            payload["stop"] = config.stop_sequences

        output_tokens = 0
        input_tokens = 0
        finish_reason = None

        with client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})

                        if delta.get("content"):
                            output_tokens += 1
                            yield StreamChunk(
                                text=delta["content"],
                                is_final=False,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                            )

                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]

                    if "usage" in data:
                        input_tokens = data["usage"].get("prompt_tokens", 0)
                        output_tokens = data["usage"].get("completion_tokens", 0)

                except json.JSONDecodeError:
                    continue

        # Final chunk
        yield StreamChunk(
            text="",
            is_final=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=self._map_finish_reason(finish_reason),
        )

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert HuggingFace messages to OpenAI format."""
        # They're essentially the same format for basic chat
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]

    def _map_finish_reason(self, reason: Optional[str]) -> str:
        """Map OpenAI finish reason to our standard format."""
        if reason is None:
            return "stop"

        mapping = {
            "stop": "stop",
            "length": "length",
            "max_tokens": "length",
            "tool_calls": "tool_calls",
            "function_call": "tool_calls",
            "content_filter": "stop",
        }
        return mapping.get(reason, "stop")

    def is_available(self) -> bool:
        """Check if the remote endpoint is reachable."""
        try:
            client = self._get_sync_client()
            response = client.get("/models", timeout=5.0)
            return response.status_code in (200, 401, 403)  # Auth errors mean server is up
        except Exception:
            return False

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

        if self._async_client:
            # For async client, we need to handle this in an async context
            # or let it be garbage collected
            self._async_client = None

        if self._openai_client:
            self._openai_client.close()
            self._openai_client = None

        logger.info("Remote backend shutdown complete")
