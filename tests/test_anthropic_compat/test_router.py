"""
Tests for Anthropic-compatible API router.

Tests:
- /v1/messages endpoint
- /v1/models endpoint
- Error handling
- Request validation
"""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from anthropic_compat.router import router, set_dependencies
from anthropic_compat.backends.base import GenerationResult, StreamChunk
from anthropic_compat.backends.config import BackendConfig, BackendType


@pytest.fixture
def mock_model_manager():
    """Create a mock model manager."""
    manager = MagicMock()

    # Setup configs
    manager.configs = {
        "test-model": MagicMock(
            type="causal_lm",
            backend_type=None,
            device="cpu",
        ),
        "embedding-model": MagicMock(
            type="embedding",
            backend_type=None,
            device="cpu",
        ),
    }

    # Mock get_model to return mock model and tokenizer
    mock_model = MagicMock()
    mock_model.config.max_position_embeddings = 4096

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.apply_chat_template.return_value = "prompt"
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.decode.return_value = "Generated text"

    manager.get_model.return_value = (mock_model, mock_tokenizer)

    return manager


@pytest.fixture
def mock_backend():
    """Create a mock inference backend."""
    backend = MagicMock()
    backend.generate.return_value = GenerationResult(
        text="Hello! I'm here to help.",
        input_tokens=10,
        output_tokens=6,
        finish_reason="stop",
    )
    return backend


@pytest.fixture
def app(mock_model_manager):
    """Create test FastAPI application."""
    app = FastAPI()
    app.include_router(router)

    # Set dependencies
    backend_config = BackendConfig(backend_type=BackendType.HUGGINGFACE)
    set_dependencies(mock_model_manager, None, backend_config)

    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestMessagesEndpoint:
    """Tests for POST /v1/messages endpoint."""

    def test_messages_basic_request(self, client, mock_backend):
        """Test basic message request."""
        with patch("anthropic_compat.router.create_backend", return_value=mock_backend):
            response = client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

        assert response.status_code == 200
        data = response.json()

        assert "id" in data
        assert data["id"].startswith("msg_")
        assert data["model"] == "test-model"
        assert data["stop_reason"] == "end_turn"
        assert "content" in data
        assert len(data["content"]) > 0
        assert data["usage"]["input_tokens"] == 10
        assert data["usage"]["output_tokens"] == 6

    def test_messages_with_system(self, client, mock_backend):
        """Test message request with system prompt."""
        with patch("anthropic_compat.router.create_backend", return_value=mock_backend):
            response = client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 100,
                    "system": "You are a helpful assistant.",
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

        assert response.status_code == 200

    def test_messages_invalid_model(self, client):
        """Test request with non-existent model."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "non-existent-model",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )

        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    def test_messages_non_llm_model(self, client):
        """Test request with embedding model (not causal_lm)."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "embedding-model",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )

        assert response.status_code == 400
        assert "not a causal language model" in response.json()["detail"]

    def test_messages_max_tokens_reached(self, client):
        """Test max_tokens stop reason."""
        mock_backend = MagicMock()
        mock_backend.generate.return_value = GenerationResult(
            text="Long response...",
            input_tokens=10,
            output_tokens=100,  # Matches max_tokens
            finish_reason="length",
        )

        with patch("anthropic_compat.router.create_backend", return_value=mock_backend):
            response = client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

        assert response.status_code == 200
        assert response.json()["stop_reason"] == "max_tokens"

    def test_messages_with_temperature(self, client, mock_backend):
        """Test temperature parameter is passed."""
        with patch("anthropic_compat.router.create_backend", return_value=mock_backend):
            response = client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 100,
                    "temperature": 0.5,
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

        assert response.status_code == 200
        # Verify generate was called
        mock_backend.generate.assert_called_once()

    def test_messages_missing_required_fields(self, client):
        """Test request missing required fields."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                # Missing max_tokens and messages
            },
        )

        assert response.status_code == 422  # Validation error


class TestModelsEndpoint:
    """Tests for GET /v1/models endpoint."""

    def test_list_models(self, client):
        """Test listing available models."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert "object" in data
        assert data["object"] == "list"

        # Should only include causal_lm models
        model_ids = [m["id"] for m in data["data"]]
        assert "test-model" in model_ids
        assert "embedding-model" not in model_ids

    def test_model_capabilities(self, client):
        """Test model capabilities in response."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        models = response.json()["data"]

        for model in models:
            assert "capabilities" in model
            assert model["capabilities"]["tool_use"] is True
            assert model["capabilities"]["streaming"] is True


class TestErrorHandling:
    """Tests for error handling."""

    def test_generation_error(self, client):
        """Test handling of generation errors."""
        mock_backend = MagicMock()
        mock_backend.generate.side_effect = RuntimeError("GPU out of memory")

        with patch("anthropic_compat.router.create_backend", return_value=mock_backend):
            response = client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

        assert response.status_code == 500
        assert "Generation failed" in response.json()["detail"]


class TestDependencyInjection:
    """Tests for dependency injection."""

    def test_model_manager_not_initialized(self):
        """Test error when model manager not set."""
        # Create fresh app without setting dependencies
        fresh_app = FastAPI()
        fresh_router = router

        # Reset global state
        import anthropic_compat.router as router_module
        original_manager = router_module._model_manager
        router_module._model_manager = None

        try:
            fresh_app.include_router(fresh_router)
            test_client = TestClient(fresh_app)

            response = test_client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

            assert response.status_code == 503
            assert "not initialized" in response.json()["detail"].lower()
        finally:
            # Restore original state
            router_module._model_manager = original_manager
