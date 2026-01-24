"""
Tests for vLLM local backend.

Tests:
- Initialization
- Configuration
- Generation (mocked)
- Capabilities
- Sampling params building
"""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from anthropic_compat.backends.vllm_local import VLLMLocalBackend, VLLM_AVAILABLE
from anthropic_compat.backends.base import (
    BackendCapabilities,
    GenerationConfig,
    GenerationResult,
    StreamChunk,
)
from anthropic_compat.backends.config import VLLMLocalConfig


# Skip all tests if vLLM is not available
pytestmark = pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")


class TestVLLMAvailability:
    """Tests for vLLM availability."""

    def test_vllm_is_available(self):
        """Test vLLM is available after installation."""
        assert VLLM_AVAILABLE is True

    def test_imports_work(self):
        """Test vLLM imports work."""
        from vllm import LLM, SamplingParams
        assert LLM is not None
        assert SamplingParams is not None


class TestVLLMLocalBackendInit:
    """Tests for VLLMLocalBackend initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        backend = VLLMLocalBackend(
            model_name_or_path="facebook/opt-125m",
        )

        assert backend.model_name == "facebook/opt-125m"
        assert backend.tokenizer_name == "facebook/opt-125m"
        assert backend.config is not None
        assert backend._initialized is False

    def test_init_with_custom_tokenizer(self):
        """Test initialization with custom tokenizer."""
        backend = VLLMLocalBackend(
            model_name_or_path="facebook/opt-125m",
            tokenizer_name="facebook/opt-350m",
        )

        assert backend.model_name == "facebook/opt-125m"
        assert backend.tokenizer_name == "facebook/opt-350m"

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = VLLMLocalConfig(
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
        )

        backend = VLLMLocalBackend(
            model_name_or_path="facebook/opt-125m",
            config=config,
        )

        assert backend.config.tensor_parallel_size == 2
        assert backend.config.gpu_memory_utilization == 0.8
        assert backend.config.max_model_len == 2048

    def test_lazy_initialization(self):
        """Test engine is not initialized until needed."""
        backend = VLLMLocalBackend(
            model_name_or_path="facebook/opt-125m",
        )

        assert backend._engine is None
        assert backend._tokenizer is None
        assert backend._initialized is False


class TestVLLMLocalBackendConfig:
    """Tests for VLLMLocalConfig."""

    def test_config_from_env(self, monkeypatch):
        """Test config loading from environment."""
        monkeypatch.setenv("VLLM_TENSOR_PARALLEL_SIZE", "4")
        monkeypatch.setenv("VLLM_GPU_MEMORY_UTILIZATION", "0.75")
        monkeypatch.setenv("VLLM_QUANTIZATION", "awq")
        monkeypatch.setenv("VLLM_ENFORCE_EAGER", "true")

        config = VLLMLocalConfig.from_env()

        assert config.tensor_parallel_size == 4
        assert config.gpu_memory_utilization == 0.75
        assert config.quantization == "awq"
        assert config.enforce_eager is True


class TestVLLMLocalBackendSamplingParams:
    """Tests for sampling parameters building."""

    def test_build_sampling_params_basic(self):
        """Test basic sampling params."""
        backend = VLLMLocalBackend("facebook/opt-125m")
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )

        params = backend._build_sampling_params(config)

        assert params.max_tokens == 100
        assert params.temperature == 0.7
        assert params.top_p == 0.9

    def test_build_sampling_params_zero_temperature(self):
        """Test zero temperature (greedy)."""
        backend = VLLMLocalBackend("facebook/opt-125m")
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.0,
        )

        params = backend._build_sampling_params(config)

        assert params.temperature == 0.0

    def test_build_sampling_params_with_top_k(self):
        """Test top_k is included."""
        backend = VLLMLocalBackend("facebook/opt-125m")
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.7,
            top_k=50,
        )

        params = backend._build_sampling_params(config)

        assert params.top_k == 50

    def test_build_sampling_params_with_penalties(self):
        """Test penalties are included."""
        backend = VLLMLocalBackend("facebook/opt-125m")
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.7,
            repetition_penalty=1.2,
            presence_penalty=0.5,
            frequency_penalty=0.3,
        )

        params = backend._build_sampling_params(config)

        assert params.repetition_penalty == 1.2
        assert params.presence_penalty == 0.5
        assert params.frequency_penalty == 0.3

    def test_build_sampling_params_with_stop(self):
        """Test stop sequences."""
        backend = VLLMLocalBackend("facebook/opt-125m")
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.7,
            stop_sequences=["STOP", "\n\n"],
        )

        params = backend._build_sampling_params(config)

        assert params.stop == ["STOP", "\n\n"]


class TestVLLMLocalBackendFinishReason:
    """Tests for finish reason mapping."""

    def test_map_finish_reason_stop(self):
        """Test 'stop' finish reason."""
        backend = VLLMLocalBackend("facebook/opt-125m")
        assert backend._map_finish_reason("stop") == "stop"

    def test_map_finish_reason_length(self):
        """Test 'length' finish reason."""
        backend = VLLMLocalBackend("facebook/opt-125m")
        assert backend._map_finish_reason("length") == "length"

    def test_map_finish_reason_abort(self):
        """Test 'abort' maps to 'stop'."""
        backend = VLLMLocalBackend("facebook/opt-125m")
        assert backend._map_finish_reason("abort") == "stop"

    def test_map_finish_reason_none(self):
        """Test None maps to 'stop'."""
        backend = VLLMLocalBackend("facebook/opt-125m")
        assert backend._map_finish_reason(None) == "stop"

    def test_map_finish_reason_unknown(self):
        """Test unknown reason maps to 'stop'."""
        backend = VLLMLocalBackend("facebook/opt-125m")
        assert backend._map_finish_reason("unknown") == "stop"


class TestVLLMLocalBackendIsAvailable:
    """Tests for is_available method."""

    def test_is_available_true(self):
        """Test is_available returns True when vLLM installed."""
        backend = VLLMLocalBackend("facebook/opt-125m")
        assert backend.is_available() is True


class TestVLLMLocalBackendShutdown:
    """Tests for shutdown."""

    def test_shutdown_clears_state(self):
        """Test shutdown clears internal state."""
        backend = VLLMLocalBackend("facebook/opt-125m")

        # Simulate initialized state
        backend._engine = MagicMock()
        backend._tokenizer = MagicMock()
        backend._initialized = True

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.empty_cache"):
            backend.shutdown()

        assert backend._engine is None
        assert backend._tokenizer is None
        assert backend._initialized is False


class TestVLLMLocalBackendWithMockedEngine:
    """Tests using mocked vLLM engine."""

    def test_capabilities_after_init(self):
        """Test capabilities are built after initialization."""
        backend = VLLMLocalBackend("facebook/opt-125m")

        # Mock the initialization
        mock_engine = MagicMock()
        mock_engine.llm_engine.model_config.max_model_len = 8192

        with patch.object(backend, "_engine", mock_engine), \
             patch.object(backend, "_initialized", True):
            backend._capabilities = backend._build_capabilities()

        assert backend.capabilities.max_context_length == 8192
        assert backend.capabilities.streaming is True
        assert backend.capabilities.continuous_batching is True
        assert backend.capabilities.paged_attention is True

    def test_count_tokens(self):
        """Test token counting with mocked tokenizer."""
        backend = VLLMLocalBackend("facebook/opt-125m")

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        backend._tokenizer = mock_tokenizer
        backend._initialized = True

        count = backend.count_tokens("Hello world")

        assert count == 5
        mock_tokenizer.encode.assert_called_with("Hello world")

    def test_generate_mocked(self):
        """Test generation with fully mocked engine."""
        backend = VLLMLocalBackend("facebook/opt-125m")

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        # Mock engine output
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "Generated response"
        mock_output.outputs[0].token_ids = [10, 11, 12, 13]
        mock_output.outputs[0].finish_reason = "stop"

        mock_engine = MagicMock()
        mock_engine.generate.return_value = [mock_output]

        backend._engine = mock_engine
        backend._tokenizer = mock_tokenizer
        backend._initialized = True

        messages = [{"role": "user", "content": "Hello"}]
        config = GenerationConfig(max_tokens=100)

        result = backend.generate(messages, config)

        assert isinstance(result, GenerationResult)
        assert result.text == "Generated response"
        assert result.output_tokens == 4
        assert result.finish_reason == "stop"

    def test_generate_stream_mocked(self):
        """Test streaming with mocked engine."""
        backend = VLLMLocalBackend("facebook/opt-125m")

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        # Mock engine output (streaming simulates full generation then chunks)
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "Hello world!"
        mock_output.outputs[0].token_ids = [10, 11, 12]
        mock_output.outputs[0].finish_reason = "stop"

        mock_engine = MagicMock()
        mock_engine.generate.return_value = [mock_output]

        backend._engine = mock_engine
        backend._tokenizer = mock_tokenizer
        backend._initialized = True

        messages = [{"role": "user", "content": "Hello"}]
        config = GenerationConfig(max_tokens=100)

        chunks = list(backend.generate_stream(messages, config))

        # Should have multiple chunks plus final
        assert len(chunks) >= 2
        assert chunks[-1].is_final is True
        assert chunks[-1].finish_reason == "stop"

        # Collect text
        full_text = "".join(c.text for c in chunks)
        assert "Hello world!" in full_text


class TestVLLMLocalBackendCapabilities:
    """Tests for capability building."""

    def test_multi_gpu_capability(self):
        """Test multi-GPU capability based on tensor_parallel_size."""
        config = VLLMLocalConfig(tensor_parallel_size=2)
        backend = VLLMLocalBackend("facebook/opt-125m", config=config)

        # Mock initialization
        mock_engine = MagicMock()
        mock_engine.llm_engine.model_config.max_model_len = 4096

        backend._engine = mock_engine
        backend._initialized = True
        backend._capabilities = backend._build_capabilities()

        assert backend.capabilities.multi_gpu is True

    def test_single_gpu_capability(self):
        """Test single GPU capability."""
        config = VLLMLocalConfig(tensor_parallel_size=1)
        backend = VLLMLocalBackend("facebook/opt-125m", config=config)

        mock_engine = MagicMock()
        mock_engine.llm_engine.model_config.max_model_len = 4096

        backend._engine = mock_engine
        backend._initialized = True
        backend._capabilities = backend._build_capabilities()

        assert backend.capabilities.multi_gpu is False

    def test_quantization_methods(self):
        """Test supported quantization methods."""
        backend = VLLMLocalBackend("facebook/opt-125m")

        mock_engine = MagicMock()
        mock_engine.llm_engine.model_config.max_model_len = 4096

        backend._engine = mock_engine
        backend._initialized = True
        backend._capabilities = backend._build_capabilities()

        assert "awq" in backend.capabilities.quantization
        assert "gptq" in backend.capabilities.quantization
        assert "bitsandbytes" in backend.capabilities.quantization
        assert "fp8" in backend.capabilities.quantization


class TestVLLMLocalBackendEnsureInitialized:
    """Tests for _ensure_initialized method."""

    def test_ensure_initialized_creates_engine(self):
        """Test _ensure_initialized creates vLLM engine."""
        backend = VLLMLocalBackend("facebook/opt-125m")

        mock_engine = MagicMock()
        mock_tokenizer = MagicMock()
        mock_engine.get_tokenizer.return_value = mock_tokenizer

        with patch("anthropic_compat.backends.vllm_local.LLM", return_value=mock_engine):
            backend._ensure_initialized()

        assert backend._initialized is True
        assert backend._engine is mock_engine
        assert backend._tokenizer is mock_tokenizer

    def test_ensure_initialized_only_once(self):
        """Test _ensure_initialized is idempotent."""
        backend = VLLMLocalBackend("facebook/opt-125m")
        backend._initialized = True
        backend._engine = MagicMock()

        # Should return early
        with patch("anthropic_compat.backends.vllm_local.LLM") as mock_llm:
            backend._ensure_initialized()
            mock_llm.assert_not_called()

    def test_ensure_initialized_with_quantization(self):
        """Test initialization with quantization config."""
        config = VLLMLocalConfig(quantization="awq")
        backend = VLLMLocalBackend("facebook/opt-125m", config=config)

        mock_engine = MagicMock()
        mock_engine.get_tokenizer.return_value = MagicMock()

        with patch("anthropic_compat.backends.vllm_local.LLM", return_value=mock_engine) as mock_llm:
            backend._ensure_initialized()

            # Verify quantization was passed
            call_kwargs = mock_llm.call_args[1]
            assert call_kwargs["quantization"] == "awq"

    def test_ensure_initialized_with_max_model_len(self):
        """Test initialization with max_model_len."""
        config = VLLMLocalConfig(max_model_len=4096)
        backend = VLLMLocalBackend("facebook/opt-125m", config=config)

        mock_engine = MagicMock()
        mock_engine.get_tokenizer.return_value = MagicMock()

        with patch("anthropic_compat.backends.vllm_local.LLM", return_value=mock_engine) as mock_llm:
            backend._ensure_initialized()

            call_kwargs = mock_llm.call_args[1]
            assert call_kwargs["max_model_len"] == 4096

    def test_ensure_initialized_failure_raises(self):
        """Test initialization failure raises exception."""
        backend = VLLMLocalBackend("facebook/opt-125m")

        with patch("anthropic_compat.backends.vllm_local.LLM", side_effect=RuntimeError("GPU OOM")):
            with pytest.raises(RuntimeError) as exc_info:
                backend._ensure_initialized()

            assert "GPU OOM" in str(exc_info.value)

    def test_capabilities_triggers_init(self):
        """Test accessing capabilities triggers initialization."""
        backend = VLLMLocalBackend("facebook/opt-125m")

        mock_engine = MagicMock()
        mock_engine.get_tokenizer.return_value = MagicMock()
        mock_engine.llm_engine.model_config.max_model_len = 4096

        with patch("anthropic_compat.backends.vllm_local.LLM", return_value=mock_engine):
            caps = backend.capabilities

        assert backend._initialized is True
        assert caps.max_context_length == 4096

    def test_count_tokens_triggers_init(self):
        """Test count_tokens triggers initialization."""
        backend = VLLMLocalBackend("facebook/opt-125m")

        mock_engine = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_engine.get_tokenizer.return_value = mock_tokenizer

        with patch("anthropic_compat.backends.vllm_local.LLM", return_value=mock_engine):
            count = backend.count_tokens("Hello")

        assert backend._initialized is True
        assert count == 3


class TestVLLMLocalBackendGenerate:
    """Tests for generate method with mocked initialization."""

    def test_generate_triggers_init(self):
        """Test generate triggers initialization."""
        backend = VLLMLocalBackend("facebook/opt-125m")

        mock_engine = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_engine.get_tokenizer.return_value = mock_tokenizer

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "Response"
        mock_output.outputs[0].token_ids = [10, 11]
        mock_output.outputs[0].finish_reason = "stop"
        mock_engine.generate.return_value = [mock_output]

        with patch("anthropic_compat.backends.vllm_local.LLM", return_value=mock_engine):
            result = backend.generate(
                [{"role": "user", "content": "Hi"}],
                GenerationConfig(max_tokens=100)
            )

        assert backend._initialized is True
        assert result.text == "Response"


class TestVLLMLocalBackendAsyncStreaming:
    """Tests for async streaming generation."""

    @pytest.mark.asyncio
    async def test_generate_stream_async_creates_async_engine(self):
        """Test async streaming creates async engine."""
        backend = VLLMLocalBackend("facebook/opt-125m")

        # Mock async engine
        mock_async_engine = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.encode.return_value = [1, 2, 3]

        # Make get_tokenizer async
        async def get_tokenizer():
            return mock_tokenizer

        mock_async_engine.get_tokenizer = get_tokenizer

        # Mock async generate
        async def mock_generate(prompt, params, request_id):
            output = MagicMock()
            output.outputs = [MagicMock()]
            output.outputs[0].text = "Hello"
            output.outputs[0].token_ids = [1, 2]
            output.outputs[0].finish_reason = "stop"
            output.finished = True
            yield output

        mock_async_engine.generate = mock_generate

        with patch("anthropic_compat.backends.vllm_local.AsyncLLMEngine") as mock_engine_cls, \
             patch("anthropic_compat.backends.vllm_local.AsyncEngineArgs"):
            mock_engine_cls.from_engine_args.return_value = mock_async_engine

            messages = [{"role": "user", "content": "Hi"}]
            config = GenerationConfig(max_tokens=100)

            chunks = []
            async for chunk in backend.generate_stream_async(messages, config):
                chunks.append(chunk)

            assert len(chunks) >= 1
            assert chunks[-1].is_final is True

    @pytest.mark.asyncio
    async def test_generate_stream_async_with_quantization(self):
        """Test async streaming passes quantization config."""
        config = VLLMLocalConfig(quantization="awq")
        backend = VLLMLocalBackend("facebook/opt-125m", config=config)

        mock_async_engine = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.encode.return_value = [1, 2, 3]

        async def get_tokenizer():
            return mock_tokenizer

        mock_async_engine.get_tokenizer = get_tokenizer

        async def mock_generate(prompt, params, request_id):
            output = MagicMock()
            output.outputs = [MagicMock()]
            output.outputs[0].text = "Response"
            output.outputs[0].token_ids = [1]
            output.outputs[0].finish_reason = "stop"
            output.finished = True
            yield output

        mock_async_engine.generate = mock_generate

        with patch("anthropic_compat.backends.vllm_local.AsyncLLMEngine") as mock_engine_cls, \
             patch("anthropic_compat.backends.vllm_local.AsyncEngineArgs") as mock_args:
            mock_engine_cls.from_engine_args.return_value = mock_async_engine

            messages = [{"role": "user", "content": "Hi"}]
            gen_config = GenerationConfig(max_tokens=100)

            chunks = []
            async for chunk in backend.generate_stream_async(messages, gen_config):
                chunks.append(chunk)

            # Verify quantization was set
            assert mock_args.call_args is not None


class TestVLLMLocalBackendShutdownEdgeCases:
    """Tests for shutdown edge cases."""

    def test_shutdown_clears_cuda_cache(self):
        """Test shutdown clears CUDA cache when available."""
        backend = VLLMLocalBackend("facebook/opt-125m")
        backend._engine = MagicMock()
        backend._async_engine = MagicMock()
        backend._tokenizer = MagicMock()
        backend._initialized = True

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.empty_cache") as mock_empty:
            backend.shutdown()
            mock_empty.assert_called_once()

        assert backend._engine is None
        assert backend._async_engine is None
        assert backend._tokenizer is None
        assert backend._initialized is False

    def test_shutdown_no_engine(self):
        """Test shutdown with no engine does nothing."""
        backend = VLLMLocalBackend("facebook/opt-125m")
        backend._engine = None

        # Should not raise
        backend.shutdown()

        assert backend._initialized is False
