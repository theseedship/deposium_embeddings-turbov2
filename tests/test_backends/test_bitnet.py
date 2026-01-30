"""
Tests for BitNet backend.

Tests:
- Configuration loading
- Backend initialization
- Generation (sync and stream)
- Error handling (timeout, model not found)
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from anthropic_compat.backends.base import GenerationConfig
from anthropic_compat.backends.bitnet import BitNetBackend
from anthropic_compat.backends.config import BackendType, BitNetConfig


class TestBitNetConfig:
    """Tests for BitNetConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BitNetConfig()

        assert config.bitnet_path == "/app/BitNet"
        assert config.model_path == "/app/models/bitnet/model.gguf"
        assert config.threads == 4
        assert config.context_length == 2048
        assert config.timeout == 120.0

    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("BITNET_PATH", "/custom/bitnet")
        monkeypatch.setenv("BITNET_MODEL_PATH", "/custom/model.gguf")
        monkeypatch.setenv("BITNET_THREADS", "8")
        monkeypatch.setenv("BITNET_CONTEXT_LENGTH", "4096")
        monkeypatch.setenv("BITNET_TIMEOUT", "60.0")

        config = BitNetConfig.from_env()

        assert config.bitnet_path == "/custom/bitnet"
        assert config.model_path == "/custom/model.gguf"
        assert config.threads == 8
        assert config.context_length == 4096
        assert config.timeout == 60.0


class TestBitNetBackend:
    """Tests for BitNetBackend."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a mock config with temporary paths."""
        return BitNetConfig(
            bitnet_path=str(tmp_path / "BitNet"),
            model_path=str(tmp_path / "model.gguf"),
            threads=4,
            context_length=2048,
            timeout=30.0,
        )

    @pytest.fixture
    def backend(self, mock_config):
        """Create a BitNet backend instance."""
        return BitNetBackend(config=mock_config)

    def test_initialization(self, backend, mock_config):
        """Test backend initializes correctly."""
        assert backend.model_path == mock_config.model_path
        assert backend.threads == mock_config.threads
        assert backend.context_length == mock_config.context_length
        assert backend.timeout == mock_config.timeout

    def test_capabilities(self, backend):
        """Test capabilities are correctly reported."""
        caps = backend.capabilities

        assert caps.streaming is True
        assert caps.tool_calling is False  # BitNet doesn't support tools
        assert caps.batching is False
        assert caps.multi_gpu is False  # CPU only
        assert "1bit" in caps.quantization
        assert caps.max_context_length == 2048

    def test_count_tokens(self, backend):
        """Test token counting estimation."""
        # ~4 chars per token
        assert backend.count_tokens("hello") == 1
        assert backend.count_tokens("hello world") == 2
        assert backend.count_tokens("a" * 100) == 25

    def test_build_prompt_simple(self, backend):
        """Test prompt building with simple messages."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        prompt = backend._build_prompt(messages)

        assert "User: Hello" in prompt
        assert prompt.endswith("Assistant:")

    def test_build_prompt_with_system(self, backend):
        """Test prompt building with system message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"}
        ]
        prompt = backend._build_prompt(messages)

        assert "System: You are helpful." in prompt
        assert "User: Hi" in prompt
        assert prompt.endswith("Assistant:")

    def test_build_prompt_conversation(self, backend):
        """Test prompt building with multi-turn conversation."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        prompt = backend._build_prompt(messages)

        assert "User: Hello" in prompt
        assert "Assistant: Hi there!" in prompt
        assert "User: How are you?" in prompt

    def test_is_available_false_when_missing(self, backend):
        """Test is_available returns False when BitNet not installed."""
        # Paths don't exist by default in test
        assert backend.is_available() is False

    def test_is_available_true_when_exists(self, backend, tmp_path):
        """Test is_available returns True when paths exist."""
        # Create mock BitNet installation
        bitnet_dir = tmp_path / "BitNet"
        bitnet_dir.mkdir()
        (bitnet_dir / "run_inference.py").touch()

        # Create mock model
        model_path = tmp_path / "model.gguf"
        model_path.touch()

        # Update backend paths
        backend.bitnet_path = bitnet_dir
        backend._inference_script = bitnet_dir / "run_inference.py"
        backend.model_path = str(model_path)

        assert backend.is_available() is True

    @patch("subprocess.run")
    def test_generate_success(self, mock_run, backend):
        """Test successful generation."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Hello! How can I help you today?",
            stderr="",
        )

        messages = [{"role": "user", "content": "Hello"}]
        config = GenerationConfig(max_tokens=100, temperature=0.7)

        result = backend.generate(messages, config)

        assert result.text == "Hello! How can I help you today?"
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.finish_reason == "stop"
        assert result.latency_ms > 0

    @patch("subprocess.run")
    def test_generate_with_stop_sequence(self, mock_run, backend):
        """Test generation stops at stop sequence."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Hello!\nUser: Something else",
            stderr="",
        )

        messages = [{"role": "user", "content": "Hello"}]
        config = GenerationConfig(max_tokens=100, stop_sequences=["User:"])

        result = backend.generate(messages, config)

        assert "User:" not in result.text
        assert result.finish_reason == "stop"

    @patch("subprocess.run")
    def test_generate_max_tokens(self, mock_run, backend):
        """Test generation respects max_tokens limit."""
        # Return a long response
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="a" * 500,
            stderr="",
        )

        messages = [{"role": "user", "content": "Hello"}]
        config = GenerationConfig(max_tokens=10)

        result = backend.generate(messages, config)

        # With ~4 chars/token, 500 chars = 125 tokens
        # Since 125 >= 10, finish_reason should be "length"
        assert result.finish_reason == "length"

    @patch("subprocess.run")
    def test_generate_error(self, mock_run, backend):
        """Test generation handles errors."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Model loading failed",
        )

        messages = [{"role": "user", "content": "Hello"}]
        config = GenerationConfig(max_tokens=100)

        with pytest.raises(RuntimeError) as exc_info:
            backend.generate(messages, config)

        assert "Model loading failed" in str(exc_info.value)

    @patch("subprocess.run")
    def test_generate_timeout(self, mock_run, backend):
        """Test generation handles timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=30.0)

        messages = [{"role": "user", "content": "Hello"}]
        config = GenerationConfig(max_tokens=100)

        with pytest.raises(TimeoutError) as exc_info:
            backend.generate(messages, config)

        assert "timeout" in str(exc_info.value).lower()

    @patch("subprocess.run")
    def test_generate_file_not_found(self, mock_run, backend):
        """Test generation handles missing BitNet."""
        mock_run.side_effect = FileNotFoundError("BitNet not found")

        messages = [{"role": "user", "content": "Hello"}]
        config = GenerationConfig(max_tokens=100)

        with pytest.raises(FileNotFoundError):
            backend.generate(messages, config)

    @patch("subprocess.Popen")
    def test_generate_stream(self, mock_popen, backend):
        """Test streaming generation."""
        # Mock stdout that yields lines
        mock_stdout = MagicMock()
        mock_stdout.readline.side_effect = [
            "Hello ",
            "world!",
            "",  # EOF
        ]

        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = MagicMock()
        mock_process.stderr.read.return_value = ""
        mock_process.returncode = 0
        mock_process.wait.return_value = None

        mock_popen.return_value = mock_process

        messages = [{"role": "user", "content": "Hi"}]
        config = GenerationConfig(max_tokens=100)

        chunks = list(backend.generate_stream(messages, config))

        # Should have content chunks + final chunk
        assert len(chunks) >= 2
        assert chunks[-1].is_final is True
        assert chunks[-1].finish_reason == "stop"

    @patch("subprocess.Popen")
    def test_generate_stream_timeout(self, mock_popen, backend):
        """Test streaming handles timeout."""
        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = ["Hello"]
        mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=30.0)

        mock_popen.return_value = mock_process

        messages = [{"role": "user", "content": "Hi"}]
        config = GenerationConfig(max_tokens=100)

        chunks = list(backend.generate_stream(messages, config))

        # Should have at least one chunk and final timeout chunk
        assert chunks[-1].is_final is True
        assert chunks[-1].finish_reason == "timeout"

    def test_parse_output_filters_status(self, backend):
        """Test output parsing filters status messages."""
        stdout = """loading model...
model loaded in 2.3s
Hello! How can I help?
tokens/s: 45.2
inference time: 1.5s"""

        result = backend._parse_output(stdout, "")

        assert "Hello! How can I help?" in result
        assert "loading model" not in result
        assert "tokens/s" not in result

    def test_shutdown(self, backend):
        """Test shutdown completes without error."""
        backend.shutdown()  # Should not raise


class TestBitNetBackendType:
    """Tests for BackendType.BITNET."""

    def test_backend_type_exists(self):
        """Test BITNET is a valid backend type."""
        assert BackendType.BITNET.value == "bitnet"

    def test_backend_type_from_string(self):
        """Test parsing 'bitnet' string."""
        assert BackendType.from_string("bitnet") == BackendType.BITNET
        assert BackendType.from_string("BITNET") == BackendType.BITNET
        assert BackendType.from_string("  bitnet  ") == BackendType.BITNET


class TestBitNetFactory:
    """Tests for BitNet backend creation via factory."""

    def test_create_backend_bitnet(self, monkeypatch, tmp_path):
        """Test creating BitNet backend via factory."""
        # Setup environment
        monkeypatch.setenv("LLM_BACKEND", "bitnet")
        monkeypatch.setenv("BITNET_PATH", str(tmp_path / "BitNet"))
        monkeypatch.setenv("BITNET_MODEL_PATH", str(tmp_path / "model.gguf"))

        from anthropic_compat.backends import BackendConfig, BackendType, create_backend

        config = BackendConfig.from_env()
        assert config.backend_type == BackendType.BITNET

        # Create backend (will fail if BitNet not installed, but tests config)
        backend = create_backend(
            str(tmp_path / "model.gguf"),
            backend_type=BackendType.BITNET,
            config=config,
        )

        assert backend is not None
        assert backend.capabilities.tool_calling is False

    def test_is_backend_available_bitnet_false(self, monkeypatch):
        """Test is_backend_available returns False for BitNet when not installed."""
        monkeypatch.setenv("BITNET_PATH", "/nonexistent/path")
        monkeypatch.setenv("BITNET_MODEL_PATH", "/nonexistent/model.gguf")

        from anthropic_compat.backends import BackendType, is_backend_available

        assert is_backend_available(BackendType.BITNET) is False
