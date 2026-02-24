"""Pydantic request/response models for all endpoints."""
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional

from ..shared import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANK_MODEL


class EmbedRequest(BaseModel):
    model: str = Field(default=DEFAULT_EMBEDDING_MODEL, description="Model to use for embeddings")
    input: Optional[str | List[str]] = Field(default=None, description="Text(s) to embed (OpenAI format)", max_length=100_000)
    prompt: Optional[str] = Field(default=None, description="Text to embed (Ollama format)", max_length=100_000)

    @model_validator(mode='after')
    def normalize_input(self):
        """Accept both 'input' (OpenAI) and 'prompt' (Ollama) formats for compatibility."""
        if self.input is None and self.prompt is None:
            raise ValueError("Either 'input' or 'prompt' must be provided")
        if self.input is None and self.prompt is not None:
            self.input = self.prompt
        # Limit batch size to prevent OOM
        if isinstance(self.input, list) and len(self.input) > 256:
            raise ValueError("Maximum 256 texts per batch")
        return self


class EmbedResponse(BaseModel):
    model: str
    embeddings: List[List[float]]


class RerankRequest(BaseModel):
    model: str = Field(default=DEFAULT_RERANK_MODEL, description="Model to use for reranking")
    query: str = Field(..., max_length=10_000)
    documents: List[str] = Field(..., max_length=1000)
    top_k: Optional[int] = Field(default=None, le=1000)

    @model_validator(mode='after')
    def validate_limits(self):
        if len(self.documents) > 1000:
            raise ValueError("Maximum 1000 documents per request")
        return self


class RerankResponse(BaseModel):
    model: str
    results: List[dict]


# 50MB base64 limit (~37MB decoded image)
MAX_BASE64_LENGTH = 50_000_000


class VisionRequest(BaseModel):
    """Request for vision-language model inference"""
    model: str = Field(default="lfm25-vl", description="Vision-language model to use")
    image: str = Field(..., description="Base64 encoded image (with or without data URI prefix)", max_length=MAX_BASE64_LENGTH)
    prompt: str = Field(default="Extract all text from this document.", description="Prompt for the model", max_length=10_000)
    max_tokens: Optional[int] = Field(default=512, description="Maximum tokens to generate", le=4096)


class VisionResponse(BaseModel):
    """Response from vision-language model"""
    model: str
    response: str
    latency_ms: float


class AudioTranscribeRequest(BaseModel):
    """Request for audio transcription via base64"""
    audio: str = Field(..., description="Base64 encoded audio (with or without data URI prefix)", max_length=MAX_BASE64_LENGTH)
    model: str = Field(default="whisper-base", description="Whisper model size: whisper-tiny, whisper-base, whisper-small, whisper-medium")
    language: Optional[str] = Field(default=None, description="ISO-639-1 language code (None for auto-detect)", max_length=5)
    task: str = Field(default="transcribe", description="'transcribe' or 'translate' (to English)")
    word_timestamps: bool = Field(default=False, description="Include word-level timestamps")


class AudioTranscribeResponse(BaseModel):
    """Response from audio transcription"""
    model: str
    text: str
    language: str
    language_probability: float
    duration: float
    segments: Optional[List[dict]] = None
    latency_ms: float


class AudioEmbedResponse(BaseModel):
    """Response from audio embedding (transcription + embedding)"""
    model: str
    text: str
    language: str
    embedding_model: str
    embeddings: List[List[float]]
    latency_ms: float


class BenchmarkRequest(BaseModel):
    """Request for running a benchmark"""
    category: str = Field(default="search", description="Benchmark category: knowledge, coding, math, reasoning, cybersecurity, search")
    provider: Optional[str] = Field(default="groq", description="LLM provider (groq, openai, anthropic)")
    model: Optional[str] = Field(default="llama-3.1-8b-instant", description="Model name")
    sample_limit: Optional[int] = Field(default=100, ge=1, le=1000, description="Maximum samples to evaluate")
    custom_corpus: Optional[List[dict]] = Field(default=None, description="Custom corpus data for search benchmarks")
