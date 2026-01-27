"""
Fixtures specific to audio transcription tests.

Provides mock audio data, mock Whisper models, and test configurations.
"""

import io
import struct
import wave
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def sample_wav_bytes():
    """
    Generate a simple WAV file in memory.

    Creates a 1-second mono audio file at 16kHz with silence.
    """
    buffer = io.BytesIO()

    # WAV parameters
    sample_rate = 16000
    num_channels = 1
    sample_width = 2  # 16-bit
    num_frames = sample_rate  # 1 second

    # Generate silence (zeros)
    audio_data = b'\x00\x00' * num_frames

    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def mock_whisper_model():
    """
    Create a mock faster-whisper model.

    Provides realistic mock behavior for transcribe() method.
    """
    model = MagicMock()

    # Mock TranscriptionInfo
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.98
    mock_info.duration = 1.0

    # Mock Segment
    mock_segment = MagicMock()
    mock_segment.id = 0
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = " Hello, this is a test."
    mock_segment.words = None

    # transcribe returns (generator, info)
    def mock_transcribe(*args, **kwargs):
        return iter([mock_segment]), mock_info

    model.transcribe.side_effect = mock_transcribe

    return model


@pytest.fixture
def mock_transcription_result():
    """Mock TranscriptionResult for testing."""
    from audio.whisper_handler import TranscriptionResult, TranscriptionSegment

    return TranscriptionResult(
        text="Hello, this is a test.",
        language="en",
        language_probability=0.98,
        duration=1.0,
        segments=[
            TranscriptionSegment(
                id=0,
                start=0.0,
                end=1.0,
                text="Hello, this is a test.",
                words=None,
            )
        ],
    )


@pytest.fixture
def mock_faster_whisper():
    """
    Patch faster-whisper WhisperModel for unit tests.

    Usage:
        def test_something(mock_faster_whisper):
            mock_model, mock_class = mock_faster_whisper
            # mock_model is the instance returned by WhisperModel()
            # mock_class is the patched WhisperModel class
    """
    mock_model = MagicMock()

    # Mock TranscriptionInfo
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.98
    mock_info.duration = 1.0

    # Mock Segment
    mock_segment = MagicMock()
    mock_segment.id = 0
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = " Hello, this is a test."
    mock_segment.words = None

    def mock_transcribe(*args, **kwargs):
        return iter([mock_segment]), mock_info

    mock_model.transcribe.side_effect = mock_transcribe

    with patch('audio.whisper_handler.WhisperModel', return_value=mock_model) as mock_class:
        yield mock_model, mock_class


@pytest.fixture
def audio_transcribe_response():
    """Expected structure of AudioTranscribeResponse."""
    return {
        "model": "whisper-base",
        "text": "Hello, this is a test.",
        "language": "en",
        "language_probability": 0.98,
        "duration": 1.0,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "Hello, this is a test.",
            }
        ],
        "latency_ms": 100.0,
    }
