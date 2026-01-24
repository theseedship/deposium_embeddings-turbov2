"""
Tests for HuggingFace backend.

Tests:
- Initialization
- Generation (sync and stream)
- Token counting
- Capabilities
- Generation kwargs building
"""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from anthropic_compat.backends.huggingface import HuggingFaceBackend
from anthropic_compat.backends.base import (
    BackendCapabilities,
    GenerationConfig,
    GenerationResult,
    StreamChunk,
)
from anthropic_compat.backends.config import HuggingFaceConfig


class TestHuggingFaceBackendInit:
    """Tests for HuggingFaceBackend initialization."""

    def test_init_with_defaults(self, mock_model, mock_tokenizer):
        """Test initialization with default config."""
        backend = HuggingFaceBackend(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        assert backend.model == mock_model
        assert backend.tokenizer == mock_tokenizer
        assert backend.device == "cpu"
        assert backend.config is not None

    def test_init_with_custom_config(self, mock_model, mock_tokenizer, hf_config):
        """Test initialization with custom config."""
        backend = HuggingFaceBackend(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
            config=hf_config,
        )

        assert backend.config == hf_config
        assert backend.config.device == "cpu"


class TestHuggingFaceBackendCapabilities:
    """Tests for backend capabilities."""

    def test_capabilities_streaming(self, mock_model, mock_tokenizer):
        """Test streaming capability."""
        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
        assert backend.capabilities.streaming is True

    def test_capabilities_tool_calling(self, mock_model, mock_tokenizer):
        """Test tool calling capability."""
        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
        assert backend.capabilities.tool_calling is True

    def test_capabilities_max_context(self, mock_model, mock_tokenizer):
        """Test max context length from model config."""
        mock_model.config.max_position_embeddings = 8192

        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")

        assert backend.capabilities.max_context_length == 8192

    def test_capabilities_no_batching(self, mock_model, mock_tokenizer):
        """Test batching is disabled for HuggingFace."""
        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
        assert backend.capabilities.batching is False

    def test_capabilities_no_paged_attention(self, mock_model, mock_tokenizer):
        """Test paged attention is disabled for HuggingFace."""
        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
        assert backend.capabilities.paged_attention is False

    def test_get_max_context_length_fallback(self, mock_tokenizer):
        """Test context length fallback when not in config."""
        # Create a mock model without max_position_embeddings
        mock_model = MagicMock()
        mock_model.config = MagicMock(spec=[])  # Empty spec means no attributes

        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")

        # Should fall back to 4096
        assert backend.get_max_context_length() == 4096


class TestHuggingFaceBackendTokens:
    """Tests for token counting."""

    def test_count_tokens(self, mock_model, mock_tokenizer):
        """Test token counting."""
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
        count = backend.count_tokens("Hello world")

        assert count == 5
        mock_tokenizer.encode.assert_called_with("Hello world", add_special_tokens=False)


class TestHuggingFaceBackendGenKwargs:
    """Tests for generation kwargs building."""

    def test_build_gen_kwargs_basic(self, mock_model, mock_tokenizer):
        """Test basic generation kwargs."""
        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )

        kwargs = backend._build_gen_kwargs(config)

        assert kwargs["max_new_tokens"] == 100
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.9
        assert kwargs["do_sample"] is True

    def test_build_gen_kwargs_zero_temperature(self, mock_model, mock_tokenizer):
        """Test zero temperature disables sampling."""
        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.0,
        )

        kwargs = backend._build_gen_kwargs(config)

        assert kwargs["do_sample"] is False
        assert kwargs["temperature"] == 1e-7  # Near-zero for numerical stability

    def test_build_gen_kwargs_top_k(self, mock_model, mock_tokenizer):
        """Test top_k is included when set."""
        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.7,
            top_k=50,
        )

        kwargs = backend._build_gen_kwargs(config)

        assert kwargs["top_k"] == 50

    def test_build_gen_kwargs_no_top_k(self, mock_model, mock_tokenizer):
        """Test top_k is not included when None."""
        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.7,
            top_k=None,
        )

        kwargs = backend._build_gen_kwargs(config)

        assert "top_k" not in kwargs

    def test_build_gen_kwargs_repetition_penalty(self, mock_model, mock_tokenizer):
        """Test repetition penalty is included when set."""
        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.7,
            repetition_penalty=1.2,
        )

        kwargs = backend._build_gen_kwargs(config)

        assert kwargs["repetition_penalty"] == 1.2

    def test_build_gen_kwargs_stop_sequences(self, mock_model, mock_tokenizer):
        """Test stop sequences are converted to token IDs."""
        mock_tokenizer.encode.return_value = [100]  # Mock token ID for stop sequence

        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.7,
            stop_sequences=["</s>", "\n\n"],
        )

        kwargs = backend._build_gen_kwargs(config)

        # Should include original eos_token_id plus stop sequence token IDs
        assert isinstance(kwargs["eos_token_id"], list)


class TestHuggingFaceBackendGenerate:
    """Tests for synchronous generation."""

    def test_generate_basic(self, mock_model, mock_tokenizer, sample_messages, default_gen_config):
        """Test basic generation."""
        # Setup mock return values
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

        # Mock tokenizer call for inputs
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = lambda self, key: MagicMock(shape=[1, 10])
        mock_tokenizer.return_value = mock_inputs

        # Mock model generate
        import torch
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
        mock_model.generate.return_value = mock_outputs

        # Mock decode
        mock_tokenizer.decode.return_value = "Generated response"

        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")

        with patch("torch.inference_mode"):
            result = backend.generate(sample_messages, default_gen_config)

        assert isinstance(result, GenerationResult)
        assert result.text == "Generated response"
        assert result.input_tokens == 10
        assert result.latency_ms is not None

    def test_generate_finish_reason_length(self, mock_model, mock_tokenizer, sample_messages):
        """Test finish reason is 'length' when hitting max tokens."""
        # Setup for 100+ output tokens
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = lambda self, key: MagicMock(shape=[1, 10])
        mock_tokenizer.return_value = mock_inputs

        import torch
        # Generate more tokens than max_tokens
        mock_outputs = torch.tensor([[i for i in range(120)]])
        mock_model.generate.return_value = mock_outputs
        mock_tokenizer.decode.return_value = "Long response"

        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
        config = GenerationConfig(max_tokens=100)

        with patch("torch.inference_mode"):
            result = backend.generate(sample_messages, config)

        assert result.finish_reason == "length"


class TestHuggingFaceBackendShutdown:
    """Tests for shutdown and cleanup."""

    def test_shutdown_clears_cuda_cache(self, mock_model, mock_tokenizer):
        """Test shutdown clears CUDA cache."""
        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cuda")

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.empty_cache") as mock_empty:
            backend.shutdown()
            mock_empty.assert_called_once()

    def test_shutdown_cpu_no_cuda_call(self, mock_model, mock_tokenizer):
        """Test shutdown on CPU doesn't call CUDA functions."""
        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")

        with patch("torch.cuda.empty_cache") as mock_empty:
            backend.shutdown()
            mock_empty.assert_not_called()


class TestHuggingFaceBackendIsAvailable:
    """Tests for is_available method."""

    def test_is_available_default_true(self, mock_model, mock_tokenizer):
        """Test is_available returns True by default."""
        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
        assert backend.is_available() is True


class TestHuggingFaceBackendContextLength:
    """Tests for context length detection."""

    def test_max_length_fallback(self, mock_tokenizer):
        """Test fallback to max_length when max_position_embeddings missing."""
        mock_model = MagicMock()
        # Only has max_length, not max_position_embeddings
        mock_model.config = MagicMock(spec=["max_length"])
        mock_model.config.max_length = 2048

        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
        assert backend.get_max_context_length() == 2048

    def test_n_positions_fallback(self, mock_tokenizer):
        """Test fallback to n_positions (GPT-2 style)."""
        mock_model = MagicMock()
        # Only has n_positions
        mock_model.config = MagicMock(spec=["n_positions"])
        mock_model.config.n_positions = 1024

        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
        assert backend.get_max_context_length() == 1024

    def test_quantization_config_detected(self, mock_tokenizer):
        """Test quantization config is detected in capabilities."""
        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 4096
        mock_model.quantization_config = MagicMock()
        mock_model.quantization_config.quant_method = "awq"

        backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")

        assert "bitsandbytes" in backend.capabilities.quantization
        assert "awq" in backend.capabilities.quantization


class TestHuggingFaceBackendStreaming:
    """Tests for streaming generation."""

    def test_generate_stream_basic(self, mock_model, mock_tokenizer, sample_messages, default_gen_config):
        """Test basic streaming generation."""
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

        # Mock tokenizer call
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = lambda self, key: MagicMock(shape=[1, 10])
        mock_tokenizer.return_value = mock_inputs

        # Mock TextIteratorStreamer - need to patch where it's imported (transformers)
        mock_streamer_instance = MagicMock()
        mock_streamer_instance.__iter__ = lambda self: iter(["Hello", " world", "!"])

        with patch("transformers.TextIteratorStreamer", return_value=mock_streamer_instance), \
             patch("torch.inference_mode"), \
             patch.object(mock_model, "generate"):

            backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
            chunks = list(backend.generate_stream(sample_messages, default_gen_config))

        # Should have text chunks plus final chunk
        assert len(chunks) >= 2
        assert chunks[-1].is_final is True

    def test_generate_stream_stop_sequence(self, mock_model, mock_tokenizer, sample_messages):
        """Test streaming stops at stop sequence."""
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = lambda self, key: MagicMock(shape=[1, 10])
        mock_tokenizer.return_value = mock_inputs

        # Streamer yields text that contains stop sequence
        mock_streamer_instance = MagicMock()
        mock_streamer_instance.__iter__ = lambda self: iter(["Hello", " STOP", " more text"])

        config = GenerationConfig(max_tokens=100, stop_sequences=["STOP"])

        with patch("transformers.TextIteratorStreamer", return_value=mock_streamer_instance), \
             patch("torch.inference_mode"), \
             patch.object(mock_model, "generate"):

            backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
            chunks = list(backend.generate_stream(sample_messages, config))

        # Should stop when encountering stop sequence
        assert any(c.is_final for c in chunks)

    def test_generate_stream_finish_reason_length(self, mock_model, mock_tokenizer, sample_messages):
        """Test streaming finish reason is 'length' when hitting max tokens."""
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = lambda self, key: MagicMock(shape=[1, 10])
        mock_tokenizer.return_value = mock_inputs

        # Generate many tokens
        mock_streamer_instance = MagicMock()
        mock_streamer_instance.__iter__ = lambda self: iter(["token"] * 150)

        config = GenerationConfig(max_tokens=100)

        with patch("transformers.TextIteratorStreamer", return_value=mock_streamer_instance), \
             patch("torch.inference_mode"), \
             patch.object(mock_model, "generate"):

            backend = HuggingFaceBackend(mock_model, mock_tokenizer, "cpu")
            chunks = list(backend.generate_stream(sample_messages, config))

        # Final chunk should have length finish reason
        final_chunk = chunks[-1]
        assert final_chunk.is_final is True
        assert final_chunk.finish_reason == "length"
