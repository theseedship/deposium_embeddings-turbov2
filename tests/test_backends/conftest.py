"""
Fixtures specific to backend tests.

Provides mock backends, fake HTTP responses, and test configurations.
"""

from unittest.mock import MagicMock, patch

import pytest

from anthropic_compat.backends.base import (
    BackendCapabilities,
    GenerationConfig,
    GenerationResult,
    StreamChunk,
)
from anthropic_compat.backends.config import (
    BackendConfig,
    BackendType,
    HuggingFaceConfig,
    RemoteOpenAIConfig,
    VLLMLocalConfig,
    VLLMRemoteConfig,
)


@pytest.fixture
def default_gen_config():
    """Default generation configuration for tests."""
    return GenerationConfig(
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        stop_sequences=None,
    )


@pytest.fixture
def mock_generation_result():
    """Mock generation result."""
    return GenerationResult(
        text="Hello, I am a helpful assistant!",
        input_tokens=15,
        output_tokens=8,
        finish_reason="stop",
        latency_ms=150.0,
        tokens_per_second=53.3,
    )


@pytest.fixture
def mock_stream_chunks():
    """Mock stream chunks for streaming tests."""
    return [
        StreamChunk(text="Hello", is_final=False, input_tokens=15, output_tokens=1),
        StreamChunk(text=", I am", is_final=False, input_tokens=15, output_tokens=3),
        StreamChunk(text=" a helpful", is_final=False, input_tokens=15, output_tokens=5),
        StreamChunk(text=" assistant!", is_final=False, input_tokens=15, output_tokens=7),
        StreamChunk(text="", is_final=True, input_tokens=15, output_tokens=7, finish_reason="stop"),
    ]


@pytest.fixture
def hf_config():
    """HuggingFace backend configuration."""
    return HuggingFaceConfig(
        device="cpu",  # Use CPU for tests
        torch_dtype="float32",
        quantization=None,
        load_in_4bit=False,
        load_in_8bit=False,
        use_flash_attention=False,
        trust_remote_code=True,
    )


@pytest.fixture
def remote_config():
    """Remote OpenAI backend configuration."""
    return RemoteOpenAIConfig(
        base_url="http://localhost:8001/v1",
        api_key="test-key",
        timeout=30.0,
        max_retries=2,
    )


@pytest.fixture
def vllm_remote_config():
    """vLLM remote backend configuration."""
    return VLLMRemoteConfig(
        base_url="http://localhost:8001/v1",
        api_key="test-key",
        timeout=30.0,
        max_retries=2,
        retry_delay=0.1,
    )


@pytest.fixture
def backend_config():
    """Full backend configuration."""
    return BackendConfig(
        backend_type=BackendType.HUGGINGFACE,
        huggingface=HuggingFaceConfig(device="cpu"),
        vllm_local=VLLMLocalConfig(),
        vllm_remote=VLLMRemoteConfig(),
        remote_openai=RemoteOpenAIConfig(),
    )


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 8,
            "total_tokens": 23,
        },
    }


@pytest.fixture
def mock_openai_stream_chunks():
    """Mock OpenAI streaming response chunks."""
    return [
        'data: {"id":"chatcmpl-123","choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n',
        'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
        'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"!"},"index":0}]}\n\n',
        'data: {"id":"chatcmpl-123","choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n',
        'data: {"id":"chatcmpl-123","usage":{"prompt_tokens":15,"completion_tokens":2}}\n\n',
        "data: [DONE]\n\n",
    ]
