"""
Tests for backend configuration module.

Tests:
- BackendType enum parsing
- Configuration dataclasses
- Environment variable loading
- Auto-detection logic
"""

import os

import pytest

from anthropic_compat.backends.config import (
    BackendConfig,
    BackendType,
    HuggingFaceConfig,
    QuantizationType,
    RemoteOpenAIConfig,
    VLLMLocalConfig,
    VLLMRemoteConfig,
)


class TestBackendType:
    """Tests for BackendType enum."""

    def test_from_string_huggingface(self):
        """Test parsing 'huggingface' string."""
        assert BackendType.from_string("huggingface") == BackendType.HUGGINGFACE

    def test_from_string_vllm_local(self):
        """Test parsing 'vllm_local' string."""
        assert BackendType.from_string("vllm_local") == BackendType.VLLM_LOCAL

    def test_from_string_vllm_remote(self):
        """Test parsing 'vllm_remote' string."""
        assert BackendType.from_string("vllm_remote") == BackendType.VLLM_REMOTE

    def test_from_string_remote_openai(self):
        """Test parsing 'remote_openai' string."""
        assert BackendType.from_string("remote_openai") == BackendType.REMOTE_OPENAI

    def test_from_string_case_insensitive(self):
        """Test case-insensitive parsing."""
        assert BackendType.from_string("HUGGINGFACE") == BackendType.HUGGINGFACE
        assert BackendType.from_string("HuggingFace") == BackendType.HUGGINGFACE

    def test_from_string_strips_whitespace(self):
        """Test whitespace is stripped."""
        assert BackendType.from_string("  huggingface  ") == BackendType.HUGGINGFACE

    def test_from_string_invalid_raises(self):
        """Test invalid backend type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            BackendType.from_string("invalid_backend")

        assert "Unknown backend type" in str(exc_info.value)
        assert "invalid_backend" in str(exc_info.value)


class TestQuantizationType:
    """Tests for QuantizationType enum."""

    def test_quantization_values(self):
        """Test all quantization types exist."""
        assert QuantizationType.NONE.value == "none"
        assert QuantizationType.BITSANDBYTES.value == "bitsandbytes"
        assert QuantizationType.AWQ.value == "awq"
        assert QuantizationType.GPTQ.value == "gptq"
        assert QuantizationType.MARLIN.value == "marlin"
        assert QuantizationType.FP8.value == "fp8"


class TestHuggingFaceConfig:
    """Tests for HuggingFaceConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HuggingFaceConfig()

        assert config.device == "cuda"
        assert config.torch_dtype == "auto"
        assert config.quantization == QuantizationType.BITSANDBYTES
        assert config.load_in_4bit is True
        assert config.load_in_8bit is False
        assert config.use_flash_attention is True
        assert config.trust_remote_code is True

    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("HF_DEVICE", "cpu")
        monkeypatch.setenv("HF_TORCH_DTYPE", "float16")
        monkeypatch.setenv("HF_QUANTIZATION", "none")
        monkeypatch.setenv("HF_LOAD_4BIT", "false")
        monkeypatch.setenv("HF_LOAD_8BIT", "true")
        monkeypatch.setenv("HF_FLASH_ATTENTION", "false")
        monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "false")

        config = HuggingFaceConfig.from_env()

        assert config.device == "cpu"
        assert config.torch_dtype == "float16"
        assert config.quantization == QuantizationType.NONE
        assert config.load_in_4bit is False
        assert config.load_in_8bit is True
        assert config.use_flash_attention is False
        assert config.trust_remote_code is False


class TestVLLMLocalConfig:
    """Tests for VLLMLocalConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VLLMLocalConfig()

        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.90
        assert config.max_model_len is None
        assert config.quantization is None
        assert config.dtype == "auto"
        assert config.enforce_eager is False
        assert config.enable_prefix_caching is True
        assert config.max_num_seqs == 256

    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("VLLM_TENSOR_PARALLEL_SIZE", "2")
        monkeypatch.setenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85")
        monkeypatch.setenv("VLLM_MAX_MODEL_LEN", "8192")
        monkeypatch.setenv("VLLM_QUANTIZATION", "awq")
        monkeypatch.setenv("VLLM_DTYPE", "half")
        monkeypatch.setenv("VLLM_ENFORCE_EAGER", "true")
        monkeypatch.setenv("VLLM_PREFIX_CACHING", "false")
        monkeypatch.setenv("VLLM_MAX_NUM_SEQS", "128")

        config = VLLMLocalConfig.from_env()

        assert config.tensor_parallel_size == 2
        assert config.gpu_memory_utilization == 0.85
        assert config.max_model_len == 8192
        assert config.quantization == "awq"
        assert config.dtype == "half"
        assert config.enforce_eager is True
        assert config.enable_prefix_caching is False
        assert config.max_num_seqs == 128


class TestVLLMRemoteConfig:
    """Tests for VLLMRemoteConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VLLMRemoteConfig()

        assert config.base_url == "http://localhost:8001/v1"
        assert config.api_key is None
        assert config.timeout == 120.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0

    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("VLLM_REMOTE_URL", "http://vllm-server:8000/v1")
        monkeypatch.setenv("VLLM_REMOTE_API_KEY", "secret-key")
        monkeypatch.setenv("VLLM_REMOTE_TIMEOUT", "60.0")
        monkeypatch.setenv("VLLM_REMOTE_MAX_RETRIES", "5")
        monkeypatch.setenv("VLLM_REMOTE_RETRY_DELAY", "2.0")

        config = VLLMRemoteConfig.from_env()

        assert config.base_url == "http://vllm-server:8000/v1"
        assert config.api_key == "secret-key"
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.retry_delay == 2.0


class TestRemoteOpenAIConfig:
    """Tests for RemoteOpenAIConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RemoteOpenAIConfig()

        assert config.base_url == "https://api.openai.com/v1"
        assert config.api_key is None
        assert config.organization is None
        assert config.timeout == 120.0
        assert config.max_retries == 3
        assert config.model_override is None

    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("REMOTE_OPENAI_URL", "https://api.together.xyz/v1")
        monkeypatch.setenv("REMOTE_OPENAI_API_KEY", "api-key")
        monkeypatch.setenv("REMOTE_OPENAI_ORG", "my-org")
        monkeypatch.setenv("REMOTE_OPENAI_TIMEOUT", "90.0")
        monkeypatch.setenv("REMOTE_OPENAI_MAX_RETRIES", "2")
        monkeypatch.setenv("REMOTE_OPENAI_MODEL", "custom-model")

        config = RemoteOpenAIConfig.from_env()

        assert config.base_url == "https://api.together.xyz/v1"
        assert config.api_key == "api-key"
        assert config.organization == "my-org"
        assert config.timeout == 90.0
        assert config.max_retries == 2
        assert config.model_override == "custom-model"


class TestBackendConfig:
    """Tests for BackendConfig master configuration."""

    def test_default_backend_type(self):
        """Test default backend type is HuggingFace."""
        config = BackendConfig()
        assert config.backend_type == BackendType.HUGGINGFACE

    def test_from_env_huggingface(self, monkeypatch):
        """Test loading HuggingFace backend from env."""
        monkeypatch.setenv("LLM_BACKEND", "huggingface")

        config = BackendConfig.from_env()

        assert config.backend_type == BackendType.HUGGINGFACE

    def test_from_env_vllm_local(self, monkeypatch):
        """Test loading vLLM local backend from env."""
        monkeypatch.setenv("LLM_BACKEND", "vllm_local")

        config = BackendConfig.from_env()

        assert config.backend_type == BackendType.VLLM_LOCAL

    def test_from_env_remote_openai(self, monkeypatch):
        """Test loading remote OpenAI backend from env."""
        monkeypatch.setenv("LLM_BACKEND", "remote_openai")

        config = BackendConfig.from_env()

        assert config.backend_type == BackendType.REMOTE_OPENAI

    def test_get_active_config_huggingface(self):
        """Test getting active config for HuggingFace."""
        config = BackendConfig(backend_type=BackendType.HUGGINGFACE)
        active = config.get_active_config()

        assert isinstance(active, HuggingFaceConfig)

    def test_get_active_config_vllm_local(self):
        """Test getting active config for vLLM local."""
        config = BackendConfig(backend_type=BackendType.VLLM_LOCAL)
        active = config.get_active_config()

        assert isinstance(active, VLLMLocalConfig)

    def test_get_active_config_remote_openai(self):
        """Test getting active config for remote OpenAI."""
        config = BackendConfig(backend_type=BackendType.REMOTE_OPENAI)
        active = config.get_active_config()

        assert isinstance(active, RemoteOpenAIConfig)

    def test_auto_detect_fallback_to_huggingface(self, monkeypatch):
        """Test auto-detection falls back to HuggingFace."""
        # Clear any env vars that might affect detection
        monkeypatch.delenv("VLLM_REMOTE_URL", raising=False)
        monkeypatch.delenv("REMOTE_OPENAI_URL", raising=False)

        detected = BackendConfig._auto_detect_backend()

        # Should be HuggingFace or VLLM_LOCAL if vllm is installed
        assert detected in (BackendType.HUGGINGFACE, BackendType.VLLM_LOCAL)

    def test_auto_detect_vllm_remote(self, monkeypatch):
        """Test auto-detection with VLLM_REMOTE_URL set."""
        monkeypatch.setenv("VLLM_REMOTE_URL", "http://localhost:8001/v1")

        detected = BackendConfig._auto_detect_backend()

        # Could be VLLM_LOCAL if vllm is installed, otherwise VLLM_REMOTE
        assert detected in (BackendType.VLLM_LOCAL, BackendType.VLLM_REMOTE)

    def test_auto_detect_remote_openai(self, monkeypatch):
        """Test auto-detection with REMOTE_OPENAI_URL set."""
        monkeypatch.delenv("VLLM_REMOTE_URL", raising=False)
        monkeypatch.setenv("REMOTE_OPENAI_URL", "https://api.together.xyz/v1")

        detected = BackendConfig._auto_detect_backend()

        # Could be VLLM_LOCAL if vllm is installed, otherwise REMOTE_OPENAI
        assert detected in (BackendType.VLLM_LOCAL, BackendType.REMOTE_OPENAI)
