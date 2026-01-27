"""
Tests for WhisperHandler audio transcription.

Tests cover:
- WhisperHandler initialization
- Model loading behavior
- Transcription functionality
- Error handling
"""

import io
from unittest.mock import MagicMock, patch

import pytest


class TestWhisperHandlerInit:
    """Tests for WhisperHandler initialization."""

    def test_init_without_faster_whisper(self):
        """Test that initialization fails gracefully without faster-whisper."""
        with patch.dict('sys.modules', {'faster_whisper': None}):
            # Force reimport to pick up the patched module
            import importlib
            from audio import whisper_handler

            # Temporarily set FASTER_WHISPER_AVAILABLE to False
            original = whisper_handler.FASTER_WHISPER_AVAILABLE
            whisper_handler.FASTER_WHISPER_AVAILABLE = False

            try:
                with pytest.raises(ImportError, match="faster-whisper not installed"):
                    whisper_handler.WhisperHandler()
            finally:
                whisper_handler.FASTER_WHISPER_AVAILABLE = original

    def test_init_with_defaults(self, mock_faster_whisper):
        """Test initialization with default parameters."""
        mock_model, mock_class = mock_faster_whisper

        from audio.whisper_handler import WhisperHandler

        handler = WhisperHandler()

        assert handler.model_size == "base"
        assert handler.device == "auto"
        assert handler.compute_type == "auto"
        assert handler._model is None  # Lazy loading

    def test_init_with_custom_model_size(self, mock_faster_whisper):
        """Test initialization with custom model size."""
        mock_model, mock_class = mock_faster_whisper

        from audio.whisper_handler import WhisperHandler

        handler = WhisperHandler(model_size="small")
        assert handler.model_size == "small"


class TestDeviceDetection:
    """Tests for device and compute type detection."""

    def test_auto_device_no_cuda(self, mock_faster_whisper):
        """Test auto device detection without CUDA."""
        mock_model, mock_class = mock_faster_whisper

        from audio.whisper_handler import WhisperHandler

        with patch('audio.whisper_handler.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            handler = WhisperHandler(device="auto", compute_type="auto")
            device, compute_type = handler._get_device_and_compute_type()

            assert device == "cpu"
            assert compute_type == "int8"  # INT8 on CPU

    def test_auto_device_with_cuda(self, mock_faster_whisper):
        """Test auto device detection with CUDA."""
        mock_model, mock_class = mock_faster_whisper

        from audio.whisper_handler import WhisperHandler

        with patch('audio.whisper_handler.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True

            handler = WhisperHandler(device="auto", compute_type="auto")
            device, compute_type = handler._get_device_and_compute_type()

            assert device == "cuda"
            assert compute_type == "float16"  # FP16 on GPU

    def test_explicit_device(self, mock_faster_whisper):
        """Test explicit device setting."""
        mock_model, mock_class = mock_faster_whisper

        from audio.whisper_handler import WhisperHandler

        handler = WhisperHandler(device="cpu", compute_type="float32")
        device, compute_type = handler._get_device_and_compute_type()

        assert device == "cpu"
        assert compute_type == "float32"


class TestModelLoading:
    """Tests for model loading behavior."""

    def test_lazy_loading(self, mock_faster_whisper):
        """Test that model is loaded lazily on first transcribe."""
        mock_model, mock_class = mock_faster_whisper

        from audio.whisper_handler import WhisperHandler

        handler = WhisperHandler()
        assert handler._model is None

        # Model should be loaded on first _load_model call
        with patch('audio.whisper_handler.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            handler._load_model()

        mock_class.assert_called_once()
        assert handler._current_model_size == "base"

    def test_model_caching(self, mock_faster_whisper):
        """Test that same model is reused from cache."""
        mock_model, mock_class = mock_faster_whisper

        from audio.whisper_handler import WhisperHandler

        with patch('audio.whisper_handler.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            handler = WhisperHandler()
            handler._load_model()
            handler._load_model()  # Second call should use cache

        # Should only be called once due to caching
        assert mock_class.call_count == 1

    def test_model_reload_on_size_change(self, mock_faster_whisper):
        """Test that model is reloaded when size changes."""
        mock_model, mock_class = mock_faster_whisper

        from audio.whisper_handler import WhisperHandler

        with patch('audio.whisper_handler.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            handler = WhisperHandler(model_size="base")
            handler._load_model()
            handler._load_model(model_size="small")

        # Should be called twice (different sizes)
        assert mock_class.call_count == 2


class TestTranscription:
    """Tests for transcription functionality."""

    def test_transcribe_bytes(self, mock_faster_whisper, sample_wav_bytes):
        """Test transcription from bytes."""
        mock_model, mock_class = mock_faster_whisper

        from audio.whisper_handler import WhisperHandler

        with patch('audio.whisper_handler.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            handler = WhisperHandler()
            result = handler.transcribe(sample_wav_bytes)

        assert result.text == "Hello, this is a test."
        assert result.language == "en"
        assert result.language_probability == 0.98
        assert len(result.segments) == 1

    def test_transcribe_with_language(self, mock_faster_whisper, sample_wav_bytes):
        """Test transcription with explicit language."""
        mock_model, mock_class = mock_faster_whisper

        from audio.whisper_handler import WhisperHandler

        with patch('audio.whisper_handler.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            handler = WhisperHandler()
            result = handler.transcribe(sample_wav_bytes, language="fr")

        # Verify language was passed to model
        mock_model.transcribe.assert_called()
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs['language'] == "fr"

    def test_transcribe_translate_task(self, mock_faster_whisper, sample_wav_bytes):
        """Test translation task."""
        mock_model, mock_class = mock_faster_whisper

        from audio.whisper_handler import WhisperHandler

        with patch('audio.whisper_handler.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            handler = WhisperHandler()
            result = handler.transcribe(sample_wav_bytes, task="translate")

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs['task'] == "translate"

    def test_transcribe_with_word_timestamps(self, mock_faster_whisper, sample_wav_bytes):
        """Test transcription with word-level timestamps."""
        mock_model, mock_class = mock_faster_whisper

        # Add mock word timestamps
        mock_word = MagicMock()
        mock_word.word = "Hello"
        mock_word.start = 0.0
        mock_word.end = 0.5
        mock_word.probability = 0.95

        mock_segment = MagicMock()
        mock_segment.id = 0
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = " Hello"
        mock_segment.words = [mock_word]

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.98
        mock_info.duration = 1.0

        mock_model.transcribe.side_effect = lambda *a, **k: (iter([mock_segment]), mock_info)

        from audio.whisper_handler import WhisperHandler

        with patch('audio.whisper_handler.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            handler = WhisperHandler()
            result = handler.transcribe(sample_wav_bytes, word_timestamps=True)

        assert result.segments[0].words is not None
        assert len(result.segments[0].words) == 1
        assert result.segments[0].words[0]['word'] == "Hello"


class TestModelInfo:
    """Tests for model info retrieval."""

    def test_get_model_info_not_loaded(self, mock_faster_whisper):
        """Test model info when model not loaded."""
        mock_model, mock_class = mock_faster_whisper

        from audio.whisper_handler import WhisperHandler

        with patch('audio.whisper_handler.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            handler = WhisperHandler(model_size="small")
            info = handler.get_model_info()

        assert info['model_size'] is None
        assert info['is_loaded'] is False
        assert "tiny" in info['available_sizes']

    def test_get_model_info_loaded(self, mock_faster_whisper):
        """Test model info when model is loaded."""
        mock_model, mock_class = mock_faster_whisper

        from audio.whisper_handler import WhisperHandler

        with patch('audio.whisper_handler.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            handler = WhisperHandler(model_size="small")
            handler._load_model()
            info = handler.get_model_info()

        assert info['model_size'] == "small"
        assert info['is_loaded'] is True


class TestUnload:
    """Tests for model unloading."""

    def test_unload_model(self, mock_faster_whisper):
        """Test model unloading frees resources."""
        mock_model, mock_class = mock_faster_whisper

        from audio.whisper_handler import WhisperHandler

        with patch('audio.whisper_handler.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            handler = WhisperHandler()
            handler._load_model()
            assert handler._model is not None

            handler.unload()
            assert handler._model is None
            assert handler._current_model_size is None


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_to_dict(self):
        """Test TranscriptionResult serialization."""
        from audio.whisper_handler import TranscriptionResult, TranscriptionSegment

        result = TranscriptionResult(
            text="Test text",
            language="en",
            language_probability=0.99,
            duration=2.5,
            segments=[
                TranscriptionSegment(
                    id=0,
                    start=0.0,
                    end=2.5,
                    text="Test text",
                    words=None,
                )
            ],
        )

        result_dict = result.to_dict()

        assert result_dict['text'] == "Test text"
        assert result_dict['language'] == "en"
        assert result_dict['language_probability'] == 0.99
        assert result_dict['duration'] == 2.5
        assert len(result_dict['segments']) == 1
        assert result_dict['segments'][0]['id'] == 0


class TestGlobalHandler:
    """Tests for global handler singleton."""

    def test_get_whisper_handler_singleton(self, mock_faster_whisper):
        """Test that get_whisper_handler returns singleton."""
        mock_model, mock_class = mock_faster_whisper

        from audio.whisper_handler import get_whisper_handler, _whisper_handler
        import audio.whisper_handler as wh

        # Reset global state
        wh._whisper_handler = None

        handler1 = get_whisper_handler()
        handler2 = get_whisper_handler()

        assert handler1 is handler2
