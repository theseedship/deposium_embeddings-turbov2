"""
Tests for RemoteOpenAI backend.

Tests:
- Initialization
- HTTP request handling with httpx
- OpenAI SDK integration (when available)
- Retry logic
- Streaming
- Error handling
"""

from unittest.mock import MagicMock, patch, AsyncMock
import json

import httpx
import pytest
import respx

from anthropic_compat.backends.remote_openai import RemoteOpenAIBackend, OPENAI_AVAILABLE
from anthropic_compat.backends.base import (
    GenerationConfig,
    GenerationResult,
    StreamChunk,
)
from anthropic_compat.backends.config import RemoteOpenAIConfig, VLLMRemoteConfig


class TestRemoteBackendInit:
    """Tests for RemoteOpenAIBackend initialization."""

    def test_init_with_remote_config(self, remote_config):
        """Test initialization with RemoteOpenAIConfig."""
        backend = RemoteOpenAIBackend(
            model_name="test-model",
            config=remote_config,
        )

        assert backend.model_name == "test-model"
        assert backend.base_url == "http://localhost:8001/v1"
        assert backend.api_key == "test-key"
        assert backend.timeout == 30.0

    def test_init_with_vllm_config(self, vllm_remote_config):
        """Test initialization with VLLMRemoteConfig."""
        backend = RemoteOpenAIBackend(
            model_name="test-model",
            config=vllm_remote_config,
        )

        assert backend.model_name == "test-model"
        assert backend.base_url == "http://localhost:8001/v1"
        assert backend.retry_delay == 0.1

    def test_init_default_config(self, monkeypatch):
        """Test initialization with default config from env."""
        monkeypatch.setenv("REMOTE_OPENAI_URL", "http://test:8000/v1")
        monkeypatch.setenv("REMOTE_OPENAI_API_KEY", "env-key")

        backend = RemoteOpenAIBackend(model_name="test-model")

        assert backend.base_url == "http://test:8000/v1"
        assert backend.api_key == "env-key"

    def test_init_model_override(self):
        """Test model name override from config."""
        config = RemoteOpenAIConfig(
            base_url="http://localhost:8001/v1",
            model_override="overridden-model",
        )

        backend = RemoteOpenAIBackend(
            model_name="original-model",
            config=config,
        )

        assert backend.model_name == "overridden-model"


class TestRemoteBackendCapabilities:
    """Tests for backend capabilities."""

    def test_capabilities(self, remote_config):
        """Test capabilities are set correctly."""
        backend = RemoteOpenAIBackend("test-model", remote_config)

        assert backend.capabilities.streaming is True
        assert backend.capabilities.tool_calling is True
        assert backend.capabilities.batching is True
        assert backend.capabilities.continuous_batching is True
        assert backend.capabilities.paged_attention is True


class TestRemoteBackendTokenCount:
    """Tests for token counting."""

    def test_count_tokens_estimate(self, remote_config):
        """Test token count estimation (4 chars per token)."""
        backend = RemoteOpenAIBackend("test-model", remote_config)

        # "Hello world" = 11 chars -> 2 tokens (floor division)
        count = backend.count_tokens("Hello world")

        assert count == 2  # 11 // 4 = 2


class TestRemoteBackendHTTPGenerate:
    """Tests for HTTP-based generation (without OpenAI SDK)."""

    @respx.mock
    def test_generate_with_httpx(self, remote_config, sample_messages, default_gen_config, mock_openai_response):
        """Test generation using raw HTTP requests."""
        respx.post("http://localhost:8001/v1/chat/completions").respond(
            json=mock_openai_response
        )

        backend = RemoteOpenAIBackend("test-model", remote_config)

        # Force httpx by mocking OPENAI_AVAILABLE
        with patch("anthropic_compat.backends.remote_openai.OPENAI_AVAILABLE", False):
            result = backend.generate(sample_messages, default_gen_config)

        assert isinstance(result, GenerationResult)
        assert result.text == "Hello! How can I help you today?"
        assert result.input_tokens == 15
        assert result.output_tokens == 8
        assert result.finish_reason == "stop"

    @respx.mock
    def test_generate_retries_on_failure(self, remote_config, sample_messages, default_gen_config, mock_openai_response):
        """Test retry logic on HTTP failure."""
        # First request fails, second succeeds
        route = respx.post("http://localhost:8001/v1/chat/completions")
        route.side_effect = [
            httpx.ConnectError("Connection refused"),
            httpx.Response(200, json=mock_openai_response),
        ]

        config = RemoteOpenAIConfig(
            base_url="http://localhost:8001/v1",
            max_retries=3,
        )
        backend = RemoteOpenAIBackend("test-model", config)

        with patch("anthropic_compat.backends.remote_openai.OPENAI_AVAILABLE", False), \
             patch("time.sleep"):  # Skip delay
            result = backend.generate(sample_messages, default_gen_config)

        assert result.text == "Hello! How can I help you today?"

    @respx.mock
    def test_generate_all_retries_fail(self, sample_messages, default_gen_config):
        """Test RuntimeError when all retries fail."""
        respx.post("http://localhost:8001/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        config = RemoteOpenAIConfig(
            base_url="http://localhost:8001/v1",
            max_retries=2,
        )
        backend = RemoteOpenAIBackend("test-model", config)

        with patch("anthropic_compat.backends.remote_openai.OPENAI_AVAILABLE", False), \
             patch("time.sleep"), \
             pytest.raises(RuntimeError) as exc_info:
            backend.generate(sample_messages, default_gen_config)

        assert "All 2 attempts failed" in str(exc_info.value)


class TestRemoteBackendMessageConversion:
    """Tests for message format conversion."""

    def test_convert_messages(self, remote_config):
        """Test HF to OpenAI message conversion."""
        backend = RemoteOpenAIBackend("test-model", remote_config)

        hf_messages = [
            {"role": "system", "content": "You are helpful", "extra": "ignored"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        converted = backend._convert_messages(hf_messages)

        assert len(converted) == 3
        assert converted[0] == {"role": "system", "content": "You are helpful"}
        assert converted[1] == {"role": "user", "content": "Hello"}
        assert converted[2] == {"role": "assistant", "content": "Hi there!"}


class TestRemoteBackendFinishReason:
    """Tests for finish reason mapping."""

    def test_map_finish_reason_stop(self, remote_config):
        """Test 'stop' finish reason."""
        backend = RemoteOpenAIBackend("test-model", remote_config)
        assert backend._map_finish_reason("stop") == "stop"

    def test_map_finish_reason_length(self, remote_config):
        """Test 'length' finish reason."""
        backend = RemoteOpenAIBackend("test-model", remote_config)
        assert backend._map_finish_reason("length") == "length"

    def test_map_finish_reason_max_tokens(self, remote_config):
        """Test 'max_tokens' maps to 'length'."""
        backend = RemoteOpenAIBackend("test-model", remote_config)
        assert backend._map_finish_reason("max_tokens") == "length"

    def test_map_finish_reason_tool_calls(self, remote_config):
        """Test 'tool_calls' finish reason."""
        backend = RemoteOpenAIBackend("test-model", remote_config)
        assert backend._map_finish_reason("tool_calls") == "tool_calls"

    def test_map_finish_reason_function_call(self, remote_config):
        """Test 'function_call' maps to 'tool_calls'."""
        backend = RemoteOpenAIBackend("test-model", remote_config)
        assert backend._map_finish_reason("function_call") == "tool_calls"

    def test_map_finish_reason_none(self, remote_config):
        """Test None maps to 'stop'."""
        backend = RemoteOpenAIBackend("test-model", remote_config)
        assert backend._map_finish_reason(None) == "stop"

    def test_map_finish_reason_unknown(self, remote_config):
        """Test unknown reason maps to 'stop'."""
        backend = RemoteOpenAIBackend("test-model", remote_config)
        assert backend._map_finish_reason("unknown_reason") == "stop"


class TestRemoteBackendIsAvailable:
    """Tests for is_available method."""

    @respx.mock
    def test_is_available_success(self, remote_config):
        """Test is_available returns True when server responds."""
        respx.get("http://localhost:8001/v1/models").respond(status_code=200)

        backend = RemoteOpenAIBackend("test-model", remote_config)
        assert backend.is_available() is True

    @respx.mock
    def test_is_available_auth_error(self, remote_config):
        """Test is_available returns True on auth errors (server is up)."""
        respx.get("http://localhost:8001/v1/models").respond(status_code=401)

        backend = RemoteOpenAIBackend("test-model", remote_config)
        assert backend.is_available() is True

    @respx.mock
    def test_is_available_connection_error(self, remote_config):
        """Test is_available returns False on connection error."""
        respx.get("http://localhost:8001/v1/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        backend = RemoteOpenAIBackend("test-model", remote_config)
        assert backend.is_available() is False


class TestRemoteBackendShutdown:
    """Tests for shutdown and cleanup."""

    def test_shutdown_closes_clients(self, remote_config):
        """Test shutdown closes HTTP clients."""
        backend = RemoteOpenAIBackend("test-model", remote_config)

        # Create mock clients
        mock_sync_client = MagicMock()
        mock_openai_client = MagicMock()

        backend._sync_client = mock_sync_client
        backend._openai_client = mock_openai_client

        backend.shutdown()

        # Verify close was called before references were cleared
        mock_sync_client.close.assert_called_once()
        mock_openai_client.close.assert_called_once()

        # Verify references are now None
        assert backend._sync_client is None
        assert backend._openai_client is None


class TestRemoteBackendStreaming:
    """Tests for streaming generation."""

    @respx.mock
    def test_stream_with_httpx(self, remote_config, sample_messages, default_gen_config):
        """Test streaming using raw HTTP requests."""
        # Mock SSE response
        sse_data = (
            'data: {"id":"1","choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
            'data: {"id":"1","choices":[{"delta":{"content":"!"},"index":0}]}\n\n'
            'data: {"id":"1","choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n'
            'data: [DONE]\n\n'
        )

        respx.post("http://localhost:8001/v1/chat/completions").respond(
            content=sse_data,
            headers={"content-type": "text/event-stream"},
        )

        backend = RemoteOpenAIBackend("test-model", remote_config)

        with patch("anthropic_compat.backends.remote_openai.OPENAI_AVAILABLE", False):
            chunks = list(backend.generate_stream(sample_messages, default_gen_config))

        # Should have text chunks plus final chunk
        assert len(chunks) >= 2
        assert chunks[-1].is_final is True
        assert chunks[-1].finish_reason == "stop"

        # Collect text from non-final chunks
        text = "".join(c.text for c in chunks if not c.is_final)
        assert "Hello" in text


class TestRemoteBackendOpenAISDK:
    """Tests for OpenAI SDK integration."""

    def test_generate_with_openai_sdk(self, remote_config, sample_messages, default_gen_config):
        """Test generation using OpenAI SDK."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from OpenAI SDK!"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client.chat.completions.create.return_value = mock_response

        backend = RemoteOpenAIBackend("test-model", remote_config)
        backend._openai_client = mock_client

        with patch("anthropic_compat.backends.remote_openai.OPENAI_AVAILABLE", True):
            result = backend.generate(sample_messages, default_gen_config)

        assert result.text == "Hello from OpenAI SDK!"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    def test_generate_with_openai_stop_sequences(self, remote_config, sample_messages):
        """Test OpenAI SDK respects stop sequences."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Stopped text"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client.chat.completions.create.return_value = mock_response

        backend = RemoteOpenAIBackend("test-model", remote_config)
        backend._openai_client = mock_client

        config = GenerationConfig(
            max_tokens=100,
            stop_sequences=["STOP", "END"],
        )

        with patch("anthropic_compat.backends.remote_openai.OPENAI_AVAILABLE", True):
            backend.generate(sample_messages, config)

        # Verify stop sequences were passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stop"] == ["STOP", "END"]

    def test_generate_with_openai_penalties(self, remote_config, sample_messages):
        """Test OpenAI SDK respects presence/frequency penalties."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client.chat.completions.create.return_value = mock_response

        backend = RemoteOpenAIBackend("test-model", remote_config)
        backend._openai_client = mock_client

        config = GenerationConfig(
            max_tokens=100,
            presence_penalty=0.5,
            frequency_penalty=0.3,
        )

        with patch("anthropic_compat.backends.remote_openai.OPENAI_AVAILABLE", True):
            backend.generate(sample_messages, config)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["presence_penalty"] == 0.5
        assert call_kwargs["frequency_penalty"] == 0.3

    def test_stream_with_openai_sdk(self, remote_config, sample_messages, default_gen_config):
        """Test streaming with OpenAI SDK."""
        mock_client = MagicMock()

        # Create mock stream chunks
        mock_chunks = []

        # First chunk with role
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].finish_reason = None
        chunk1.usage = None
        mock_chunks.append(chunk1)

        # Second chunk with content
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"
        chunk2.choices[0].finish_reason = None
        chunk2.usage = None
        mock_chunks.append(chunk2)

        # Final chunk with finish reason
        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = None
        chunk3.choices[0].finish_reason = "stop"
        chunk3.usage = MagicMock()
        chunk3.usage.prompt_tokens = 10
        chunk3.usage.completion_tokens = 2
        mock_chunks.append(chunk3)

        mock_client.chat.completions.create.return_value = iter(mock_chunks)

        backend = RemoteOpenAIBackend("test-model", remote_config)
        backend._openai_client = mock_client

        with patch("anthropic_compat.backends.remote_openai.OPENAI_AVAILABLE", True):
            chunks = list(backend.generate_stream(sample_messages, default_gen_config))

        # Should have content chunks plus final
        assert len(chunks) >= 2
        assert chunks[-1].is_final is True

    def test_openai_client_not_installed(self, remote_config):
        """Test error when OpenAI SDK required but not installed."""
        backend = RemoteOpenAIBackend("test-model", remote_config)

        with patch("anthropic_compat.backends.remote_openai.OPENAI_AVAILABLE", False):
            with pytest.raises(ImportError) as exc_info:
                backend._get_openai_client()

            assert "OpenAI SDK is not installed" in str(exc_info.value)


class TestRemoteBackendClientManagement:
    """Tests for HTTP client management."""

    def test_get_sync_client_creates_once(self, remote_config):
        """Test sync client is created only once."""
        backend = RemoteOpenAIBackend("test-model", remote_config)

        client1 = backend._get_sync_client()
        client2 = backend._get_sync_client()

        assert client1 is client2

    def test_get_async_client_creates_once(self, remote_config):
        """Test async client is created only once."""
        backend = RemoteOpenAIBackend("test-model", remote_config)

        client1 = backend._get_async_client()
        client2 = backend._get_async_client()

        assert client1 is client2

    def test_client_headers_set(self, remote_config):
        """Test client has correct authorization headers."""
        backend = RemoteOpenAIBackend("test-model", remote_config)
        client = backend._get_sync_client()

        assert "Authorization" in client.headers
        assert "Bearer test-key" in client.headers["Authorization"]
