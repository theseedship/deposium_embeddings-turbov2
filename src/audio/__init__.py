"""
Audio processing module for Deposium Embeddings.

Provides:
- WhisperHandler: Audio transcription using faster-whisper
- Audio embedding pipeline (transcribe + embed)
"""

from .whisper_handler import WhisperHandler, get_whisper_handler

__all__ = ["WhisperHandler", "get_whisper_handler"]
