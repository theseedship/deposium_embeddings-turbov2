"""
Tests for audio transcription API endpoints.

Tests cover:
- /api/transcribe (file upload)
- /api/transcribe/base64
- /api/audio/embed
- Error handling and validation
"""

import base64
import io
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_whisper_available():
    """Patch WHISPER_AVAILABLE to True for endpoint tests."""
    with patch('main.WHISPER_AVAILABLE', True):
        yield


@pytest.fixture
def mock_whisper_handler():
    """Create mock whisper handler for endpoint tests."""
    mock_handler = MagicMock()

    # Mock TranscriptionResult
    mock_result = MagicMock()
    mock_result.text = "Hello, this is a test transcription."
    mock_result.language = "en"
    mock_result.language_probability = 0.98
    mock_result.duration = 5.0
    mock_result.segments = []

    # Mock segment
    mock_segment = MagicMock()
    mock_segment.to_dict.return_value = {
        "id": 0,
        "start": 0.0,
        "end": 5.0,
        "text": "Hello, this is a test transcription.",
    }
    mock_result.segments = [mock_segment]

    mock_handler.transcribe.return_value = mock_result

    return mock_handler


@pytest.fixture
def test_app(mock_whisper_available, mock_whisper_handler):
    """Create test FastAPI app with mocked dependencies."""
    with patch('main.get_whisper_handler', return_value=mock_whisper_handler):
        with patch('main.verify_api_key', return_value="test-key"):
            from main import app
            yield TestClient(app)


class TestTranscribeFileEndpoint:
    """Tests for /api/transcribe endpoint."""

    def test_transcribe_file_success(self, test_app, sample_wav_bytes, mock_whisper_handler):
        """Test successful file transcription."""
        with patch('main.get_whisper_handler', return_value=mock_whisper_handler):
            response = test_app.post(
                "/api/transcribe",
                files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
                data={"model": "whisper-base"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "language" in data
        assert "latency_ms" in data
        assert data["model"] == "whisper-base"

    def test_transcribe_file_with_language(self, test_app, sample_wav_bytes, mock_whisper_handler):
        """Test transcription with specified language."""
        with patch('main.get_whisper_handler', return_value=mock_whisper_handler):
            response = test_app.post(
                "/api/transcribe",
                files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
                data={"model": "whisper-base", "language": "fr"},
            )

        assert response.status_code == 200
        mock_whisper_handler.transcribe.assert_called()
        call_kwargs = mock_whisper_handler.transcribe.call_args[1]
        assert call_kwargs.get('language') == "fr"

    def test_transcribe_file_invalid_model(self, test_app, sample_wav_bytes):
        """Test transcription with invalid model."""
        response = test_app.post(
            "/api/transcribe",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"model": "invalid-model"},
        )

        assert response.status_code == 400
        assert "Invalid model" in response.json()["detail"]

    def test_transcribe_file_no_file(self, test_app):
        """Test transcription without file."""
        response = test_app.post(
            "/api/transcribe",
            data={"model": "whisper-base"},
        )

        assert response.status_code == 422  # Validation error


class TestTranscribeBase64Endpoint:
    """Tests for /api/transcribe/base64 endpoint."""

    def test_transcribe_base64_success(self, test_app, sample_wav_bytes, mock_whisper_handler):
        """Test successful base64 transcription."""
        audio_b64 = base64.b64encode(sample_wav_bytes).decode()

        with patch('main.get_whisper_handler', return_value=mock_whisper_handler):
            response = test_app.post(
                "/api/transcribe/base64",
                json={
                    "audio": audio_b64,
                    "model": "whisper-base",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert data["model"] == "whisper-base"

    def test_transcribe_base64_with_data_uri(self, test_app, sample_wav_bytes, mock_whisper_handler):
        """Test base64 transcription with data URI prefix."""
        audio_b64 = base64.b64encode(sample_wav_bytes).decode()
        data_uri = f"data:audio/wav;base64,{audio_b64}"

        with patch('main.get_whisper_handler', return_value=mock_whisper_handler):
            response = test_app.post(
                "/api/transcribe/base64",
                json={
                    "audio": data_uri,
                    "model": "whisper-base",
                },
            )

        assert response.status_code == 200

    def test_transcribe_base64_invalid_model(self, test_app, sample_wav_bytes):
        """Test base64 transcription with invalid model."""
        audio_b64 = base64.b64encode(sample_wav_bytes).decode()

        response = test_app.post(
            "/api/transcribe/base64",
            json={
                "audio": audio_b64,
                "model": "invalid-model",
            },
        )

        assert response.status_code == 400

    def test_transcribe_base64_word_timestamps(self, test_app, sample_wav_bytes, mock_whisper_handler):
        """Test base64 transcription with word timestamps."""
        audio_b64 = base64.b64encode(sample_wav_bytes).decode()

        with patch('main.get_whisper_handler', return_value=mock_whisper_handler):
            response = test_app.post(
                "/api/transcribe/base64",
                json={
                    "audio": audio_b64,
                    "model": "whisper-base",
                    "word_timestamps": True,
                },
            )

        assert response.status_code == 200
        call_kwargs = mock_whisper_handler.transcribe.call_args[1]
        assert call_kwargs.get('word_timestamps') is True


class TestAudioEmbedEndpoint:
    """Tests for /api/audio/embed endpoint."""

    @pytest.fixture
    def mock_model_manager(self):
        """Create mock model manager for embed tests."""
        mock_manager = MagicMock()
        mock_manager.configs = {
            "m2v-bge-m3-1024d": MagicMock(),
        }

        # Mock embedding model
        mock_embed_model = MagicMock()
        mock_embed_model.encode.return_value = [[0.1] * 1024]  # 1024D embeddings
        mock_embed_model._truncate_dims = None

        mock_manager.get_model.return_value = mock_embed_model

        return mock_manager

    def test_audio_embed_success(
        self, test_app, sample_wav_bytes, mock_whisper_handler, mock_model_manager
    ):
        """Test successful audio embedding pipeline."""
        with patch('main.get_whisper_handler', return_value=mock_whisper_handler):
            with patch('main.model_manager', mock_model_manager):
                response = test_app.post(
                    "/api/audio/embed",
                    files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
                    data={
                        "whisper_model": "whisper-base",
                        "embedding_model": "m2v-bge-m3-1024d",
                    },
                )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "embeddings" in data
        assert "embedding_model" in data
        assert data["embedding_model"] == "m2v-bge-m3-1024d"

    def test_audio_embed_invalid_whisper_model(self, test_app, sample_wav_bytes, mock_model_manager):
        """Test audio embedding with invalid whisper model."""
        with patch('main.model_manager', mock_model_manager):
            response = test_app.post(
                "/api/audio/embed",
                files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
                data={
                    "whisper_model": "invalid-whisper",
                    "embedding_model": "m2v-bge-m3-1024d",
                },
            )

        assert response.status_code == 400

    def test_audio_embed_invalid_embedding_model(
        self, test_app, sample_wav_bytes, mock_whisper_handler
    ):
        """Test audio embedding with invalid embedding model."""
        mock_manager = MagicMock()
        mock_manager.configs = {}  # No models configured

        with patch('main.get_whisper_handler', return_value=mock_whisper_handler):
            with patch('main.model_manager', mock_manager):
                response = test_app.post(
                    "/api/audio/embed",
                    files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
                    data={
                        "whisper_model": "whisper-base",
                        "embedding_model": "nonexistent-model",
                    },
                )

        assert response.status_code == 400


class TestWhisperNotAvailable:
    """Tests for when Whisper is not available."""

    def test_transcribe_whisper_not_available(self, sample_wav_bytes):
        """Test transcribe endpoint when Whisper not installed."""
        with patch('main.WHISPER_AVAILABLE', False):
            with patch('main.verify_api_key', return_value="test-key"):
                from main import app
                client = TestClient(app)

                response = client.post(
                    "/api/transcribe",
                    files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
                )

        assert response.status_code == 501
        assert "faster-whisper" in response.json()["detail"]


class TestModelSizeParsing:
    """Tests for model size parsing in endpoints."""

    def test_model_with_prefix(self, test_app, sample_wav_bytes, mock_whisper_handler):
        """Test that 'whisper-base' is parsed correctly."""
        with patch('main.get_whisper_handler', return_value=mock_whisper_handler):
            response = test_app.post(
                "/api/transcribe",
                files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
                data={"model": "whisper-base"},
            )

        assert response.status_code == 200
        assert response.json()["model"] == "whisper-base"

    def test_model_all_sizes(self, test_app, sample_wav_bytes, mock_whisper_handler):
        """Test all valid model sizes."""
        valid_models = ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium"]

        for model in valid_models:
            with patch('main.get_whisper_handler', return_value=mock_whisper_handler):
                response = test_app.post(
                    "/api/transcribe",
                    files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
                    data={"model": model},
                )

            assert response.status_code == 200, f"Failed for model {model}"
