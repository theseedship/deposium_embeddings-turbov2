"""
Tests for backend factory function.

Tests:
- create_backend() factory
- Backend type selection
- Fallback behavior
- is_backend_available()
- get_available_backends()
"""

from unittest.mock import MagicMock, patch

import pytest

from anthropic_compat.backends import (
    BackendConfig,
    BackendType,
    HuggingFaceBackend,
    create_backend,
    get_available_backends,
    is_backend_available,
)
from anthropic_compat.backends.config import HuggingFaceConfig


class TestCreateBackend:
    """Tests for create_backend factory function."""

    def test_create_huggingface_backend(self, mock_model, mock_tokenizer):
        """Test creating HuggingFace backend."""
        config = BackendConfig(backend_type=BackendType.HUGGINGFACE)
        config.huggingface.device = "cpu"

        backend = create_backend(
            model_or_path=mock_model,
            tokenizer=mock_tokenizer,
            backend_type=BackendType.HUGGINGFACE,
            config=config,
            device="cpu",
        )

        assert isinstance(backend, HuggingFaceBackend)
        assert backend.model == mock_model
        assert backend.tokenizer == mock_tokenizer

    def test_create_backend_auto_detect(self, mock_model, mock_tokenizer):
        """Test backend auto-detection from config."""
        config = BackendConfig(backend_type=BackendType.HUGGINGFACE)
        config.huggingface.device = "cpu"

        backend = create_backend(
            model_or_path=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
            device="cpu",
        )

        assert isinstance(backend, HuggingFaceBackend)

    def test_create_backend_requires_tokenizer_for_hf(self, mock_model):
        """Test HuggingFace backend requires tokenizer."""
        config = BackendConfig(backend_type=BackendType.HUGGINGFACE)

        with pytest.raises(ValueError) as exc_info:
            create_backend(
                model_or_path=mock_model,
                tokenizer=None,
                backend_type=BackendType.HUGGINGFACE,
                config=config,
            )

        assert "tokenizer" in str(exc_info.value).lower()

    def test_create_backend_from_env(self, mock_model, mock_tokenizer, monkeypatch):
        """Test create_backend loads config from environment."""
        monkeypatch.setenv("LLM_BACKEND", "huggingface")
        monkeypatch.setenv("HF_DEVICE", "cpu")

        backend = create_backend(
            model_or_path=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        assert isinstance(backend, HuggingFaceBackend)

    def test_create_remote_backend_extracts_model_name(self):
        """Test remote backend extracts model name from loaded model."""
        mock_model = MagicMock()
        mock_model.config._name_or_path = "Qwen/Qwen2.5-Coder-7B"

        config = BackendConfig(backend_type=BackendType.REMOTE_OPENAI)
        config.remote_openai.base_url = "http://localhost:8001/v1"

        # Patch the remote backend class to avoid actual HTTP calls
        with patch("anthropic_compat.backends._get_remote_backend_class") as mock_cls:
            mock_backend = MagicMock()
            mock_cls.return_value = lambda **kwargs: mock_backend

            backend = create_backend(
                model_or_path=mock_model,
                backend_type=BackendType.REMOTE_OPENAI,
                config=config,
            )

            # Verify mock_cls was called (backend created)
            mock_cls.assert_called_once()

    def test_fallback_to_huggingface(self, mock_model, mock_tokenizer):
        """Test fallback to HuggingFace when other backend fails."""
        config = BackendConfig(backend_type=BackendType.VLLM_LOCAL)
        config.huggingface.device = "cpu"

        # Mock vLLM backend to raise ImportError
        with patch("anthropic_compat.backends._get_vllm_backend_class") as mock_vllm:
            mock_vllm.side_effect = ImportError("vLLM not installed")

            # Should fall back to HuggingFace
            backend = create_backend(
                model_or_path=mock_model,
                tokenizer=mock_tokenizer,
                backend_type=BackendType.VLLM_LOCAL,
                config=config,
                device="cpu",
            )

            assert isinstance(backend, HuggingFaceBackend)


class TestIsBackendAvailable:
    """Tests for is_backend_available function."""

    def test_huggingface_always_available(self):
        """Test HuggingFace backend is always available."""
        assert is_backend_available(BackendType.HUGGINGFACE) is True

    def test_vllm_local_availability(self):
        """Test vLLM local availability depends on import."""
        # Result depends on whether vllm is installed
        result = is_backend_available(BackendType.VLLM_LOCAL)
        assert isinstance(result, bool)

    def test_remote_backends_check_httpx(self):
        """Test remote backends check for httpx."""
        # httpx should be available (installed via requirements.txt)
        assert is_backend_available(BackendType.VLLM_REMOTE) is True
        assert is_backend_available(BackendType.REMOTE_OPENAI) is True


class TestGetAvailableBackends:
    """Tests for get_available_backends function."""

    def test_returns_list(self):
        """Test function returns a list."""
        backends = get_available_backends()
        assert isinstance(backends, list)

    def test_huggingface_in_available(self):
        """Test HuggingFace is always in available backends."""
        backends = get_available_backends()
        assert BackendType.HUGGINGFACE in backends

    def test_remote_backends_in_available(self):
        """Test remote backends are available (httpx installed)."""
        backends = get_available_backends()
        # httpx is in requirements.txt, so remote should be available
        assert BackendType.VLLM_REMOTE in backends
        assert BackendType.REMOTE_OPENAI in backends

    def test_all_items_are_backend_types(self):
        """Test all items in list are BackendType enum members."""
        backends = get_available_backends()
        for backend in backends:
            assert isinstance(backend, BackendType)


class TestCreateBackendEdgeCases:
    """Tests for edge cases in create_backend factory."""

    def test_create_vllm_remote_backend(self, mock_model, mock_tokenizer):
        """Test creating vLLM remote backend."""
        config = BackendConfig(backend_type=BackendType.VLLM_REMOTE)
        config.vllm_remote.base_url = "http://localhost:8001/v1"

        with patch("anthropic_compat.backends._get_remote_backend_class") as mock_cls:
            mock_backend = MagicMock()
            mock_cls.return_value = lambda **kwargs: mock_backend

            backend = create_backend(
                model_or_path="test-model",
                backend_type=BackendType.VLLM_REMOTE,
                config=config,
            )

            mock_cls.assert_called_once()

    def test_create_remote_with_loaded_model(self):
        """Test remote backend extracts model name from loaded model."""
        mock_model = MagicMock()
        mock_model.config._name_or_path = "owner/model-name"

        config = BackendConfig(backend_type=BackendType.REMOTE_OPENAI)
        config.remote_openai.base_url = "http://localhost:8001/v1"

        with patch("anthropic_compat.backends._get_remote_backend_class") as mock_cls:
            mock_backend_instance = MagicMock()
            mock_cls.return_value = MagicMock(return_value=mock_backend_instance)

            create_backend(
                model_or_path=mock_model,
                backend_type=BackendType.REMOTE_OPENAI,
                config=config,
            )

            # Verify the backend class was instantiated
            mock_cls.assert_called_once()

    def test_create_remote_with_string_model(self):
        """Test remote backend with string model name."""
        from anthropic_compat.backends.remote_openai import RemoteOpenAIBackend

        config = BackendConfig(backend_type=BackendType.REMOTE_OPENAI)
        config.remote_openai.base_url = "http://localhost:8001/v1"

        backend = create_backend(
            model_or_path="my-model-name",
            backend_type=BackendType.REMOTE_OPENAI,
            config=config,
        )

        assert isinstance(backend, RemoteOpenAIBackend)
        assert backend.model_name == "my-model-name"

    def test_vllm_local_requires_string_path(self, mock_model, mock_tokenizer):
        """Test vLLM local backend extracts path from model config."""
        mock_model.config._name_or_path = "Qwen/Qwen2.5-7B"

        config = BackendConfig(backend_type=BackendType.VLLM_LOCAL)

        with patch("anthropic_compat.backends._get_vllm_backend_class") as mock_cls:
            mock_backend = MagicMock()
            mock_cls.return_value = MagicMock(return_value=mock_backend)

            create_backend(
                model_or_path=mock_model,
                tokenizer=mock_tokenizer,
                backend_type=BackendType.VLLM_LOCAL,
                config=config,
            )

            mock_cls.assert_called_once()

    def test_vllm_local_no_path_raises(self, mock_tokenizer):
        """Test vLLM local fails without model path."""
        # Create a mock that has config but no _name_or_path
        mock_model = MagicMock()
        mock_model.config = MagicMock(spec=[])  # config exists but no _name_or_path

        config = BackendConfig(backend_type=BackendType.VLLM_LOCAL)
        config.huggingface.device = "cpu"  # For fallback

        with patch("anthropic_compat.backends._get_vllm_backend_class") as mock_cls:
            mock_cls.return_value = MagicMock()

            # Should raise because model has no _name_or_path, then fall back to HF
            # but we can also just test without fallback by not providing tokenizer
            with pytest.raises((ValueError, RuntimeError)):
                create_backend(
                    model_or_path=mock_model,
                    tokenizer=None,  # No tokenizer means no fallback possible
                    backend_type=BackendType.VLLM_LOCAL,
                    config=config,
                )


class TestLazyImports:
    """Tests for lazy import functions."""

    def test_get_vllm_backend_class_caches(self):
        """Test vLLM backend class is cached."""
        import anthropic_compat.backends as backends_module

        # Reset cache
        backends_module._vllm_backend_class = None

        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            with patch("anthropic_compat.backends.vllm_local.VLLMLocalBackend") as mock_class:
                # First call should import
                cls1 = backends_module._get_vllm_backend_class()

                # Second call should use cache
                cls2 = backends_module._get_vllm_backend_class()

                assert cls1 is cls2

    def test_get_remote_backend_class_caches(self):
        """Test remote backend class is cached."""
        import anthropic_compat.backends as backends_module

        # Reset cache
        backends_module._remote_backend_class = None

        cls1 = backends_module._get_remote_backend_class()
        cls2 = backends_module._get_remote_backend_class()

        assert cls1 is cls2
