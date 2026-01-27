"""
Whisper Audio Transcription Handler
====================================

Uses faster-whisper for 4x faster inference than OpenAI Whisper.
Supports CPU-only inference for deployment on Railway/low-memory servers.

Models:
- whisper-tiny: Fastest, ~40MB, 7.8% WER
- whisper-base: Balanced, ~1GB RAM, 5.0% WER (default)
- whisper-small: Better accuracy, ~2GB RAM, 3.4% WER
- whisper-medium: High accuracy, ~5GB RAM, 2.9% WER
- whisper-large-v3: Best accuracy, ~10GB RAM, 2.5% WER

Features:
- Automatic language detection
- Word-level timestamps
- VAD (Voice Activity Detection) for better accuracy
- Lazy model loading with caching
"""

import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass
import io

logger = logging.getLogger(__name__)

# Check for faster-whisper availability
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None
    FASTER_WHISPER_AVAILABLE = False


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio with timestamps."""
    id: int
    start: float  # Start time in seconds
    end: float    # End time in seconds
    text: str
    words: Optional[List[Dict[str, Any]]] = None  # Word-level timestamps

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "text": self.text,
        }
        if self.words:
            result["words"] = self.words
        return result


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    text: str
    language: str
    language_probability: float
    duration: float  # Audio duration in seconds
    segments: List[TranscriptionSegment]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "language": self.language,
            "language_probability": round(self.language_probability, 3),
            "duration": round(self.duration, 2),
            "segments": [s.to_dict() for s in self.segments],
        }


class WhisperHandler:
    """
    Handler for Whisper audio transcription.

    Uses faster-whisper (CTranslate2) for optimized inference.
    Supports lazy loading and model caching.
    """

    # Model size to RAM/VRAM requirements (approximate)
    MODEL_SIZES = {
        "tiny": {"ram_mb": 150, "vram_mb": 1000},
        "base": {"ram_mb": 300, "vram_mb": 1500},
        "small": {"ram_mb": 500, "vram_mb": 2500},
        "medium": {"ram_mb": 1500, "vram_mb": 5000},
        "large-v3": {"ram_mb": 3000, "vram_mb": 10000},
    }

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        download_root: Optional[str] = None,
    ):
        """
        Initialize WhisperHandler.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: "cuda", "cpu", or "auto" (auto-detect)
            compute_type: Quantization type ("int8", "float16", "float32", "auto")
            download_root: Directory to download models to
        """
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper not installed. Install with: pip install faster-whisper"
            )

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self._model: Optional[WhisperModel] = None
        self._current_model_size: Optional[str] = None

    def _get_device_and_compute_type(self) -> Tuple[str, str]:
        """
        Determine optimal device and compute type.

        Returns:
            (device, compute_type)
        """
        import torch

        device = self.device
        compute_type = self.compute_type

        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Auto-detect compute type based on device
        if compute_type == "auto":
            if device == "cuda":
                # Use float16 on GPU for best speed/quality tradeoff
                compute_type = "float16"
            else:
                # Use int8 on CPU for speed
                compute_type = "int8"

        return device, compute_type

    def _load_model(self, model_size: Optional[str] = None) -> WhisperModel:
        """
        Load or reload Whisper model.

        Args:
            model_size: Optional model size override

        Returns:
            Loaded WhisperModel
        """
        target_size = model_size or self.model_size

        # Return cached model if same size
        if self._model is not None and self._current_model_size == target_size:
            return self._model

        # Release old model
        if self._model is not None:
            logger.info(f"Unloading whisper-{self._current_model_size}")
            del self._model
            self._model = None
            import gc
            gc.collect()

        device, compute_type = self._get_device_and_compute_type()

        logger.info(f"Loading whisper-{target_size} (device={device}, compute={compute_type})")

        start_time = time.time()

        # Load model with faster-whisper
        self._model = WhisperModel(
            target_size,
            device=device,
            compute_type=compute_type,
            download_root=self.download_root,
        )

        load_time = time.time() - start_time
        self._current_model_size = target_size

        logger.info(f"whisper-{target_size} loaded in {load_time:.1f}s")

        return self._model

    def transcribe(
        self,
        audio: Union[str, bytes, io.BytesIO],
        language: Optional[str] = None,
        task: str = "transcribe",  # or "translate" (to English)
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = False,
        model_size: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio file.

        Args:
            audio: Audio file path, bytes, or BytesIO
            language: ISO-639-1 language code (None for auto-detect)
            task: "transcribe" or "translate" (translate to English)
            beam_size: Beam search size (higher = more accurate but slower)
            vad_filter: Use VAD to filter non-speech (recommended)
            word_timestamps: Include word-level timestamps
            model_size: Override model size for this transcription

        Returns:
            TranscriptionResult with text, segments, and metadata
        """
        # Load model (uses cache if same size)
        model = self._load_model(model_size)

        # Prepare audio input
        if isinstance(audio, bytes):
            audio = io.BytesIO(audio)

        start_time = time.time()

        # Transcribe with faster-whisper
        segments_generator, info = model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
        )

        # Collect segments
        segments = []
        full_text_parts = []

        for i, segment in enumerate(segments_generator):
            # Extract word timestamps if available
            words = None
            if word_timestamps and hasattr(segment, 'words') and segment.words:
                words = [
                    {
                        "word": w.word,
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                        "probability": round(w.probability, 3),
                    }
                    for w in segment.words
                ]

            seg = TranscriptionSegment(
                id=i,
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                words=words,
            )
            segments.append(seg)
            full_text_parts.append(segment.text.strip())

        transcription_time = time.time() - start_time

        # Combine full text
        full_text = " ".join(full_text_parts)

        # Log performance
        audio_duration = info.duration if hasattr(info, 'duration') else 0
        rtf = transcription_time / audio_duration if audio_duration > 0 else 0
        logger.info(
            f"Transcribed {audio_duration:.1f}s audio in {transcription_time:.1f}s "
            f"(RTF: {rtf:.2f}x, language: {info.language})"
        )

        return TranscriptionResult(
            text=full_text,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration if hasattr(info, 'duration') else 0,
            segments=segments,
        )

    def transcribe_file(
        self,
        file_path: str,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe audio from file path.

        Args:
            file_path: Path to audio file
            **kwargs: Additional arguments passed to transcribe()

        Returns:
            TranscriptionResult
        """
        return self.transcribe(file_path, **kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about current model state.

        Returns:
            Dictionary with model info
        """
        device, compute_type = self._get_device_and_compute_type()

        return {
            "model_size": self._current_model_size,
            "is_loaded": self._model is not None,
            "device": device,
            "compute_type": compute_type,
            "available_sizes": list(self.MODEL_SIZES.keys()),
        }

    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            logger.info(f"Unloading whisper-{self._current_model_size}")
            del self._model
            self._model = None
            self._current_model_size = None

            import gc
            gc.collect()

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Global instance
_whisper_handler: Optional[WhisperHandler] = None


def get_whisper_handler(
    model_size: str = "base",
    device: str = "auto",
    compute_type: str = "auto",
) -> WhisperHandler:
    """
    Get global WhisperHandler instance.

    Creates handler on first call, reuses on subsequent calls.

    Args:
        model_size: Whisper model size
        device: Device to use
        compute_type: Compute type for quantization

    Returns:
        WhisperHandler instance
    """
    global _whisper_handler

    if _whisper_handler is None:
        _whisper_handler = WhisperHandler(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
        )

    return _whisper_handler
