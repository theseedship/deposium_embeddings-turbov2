"""Audio transcription and embedding routes."""
import base64
import logging
import time
from typing import Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from .. import shared
from ..schemas.requests import AudioTranscribeRequest, AudioTranscribeResponse, AudioEmbedResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/transcribe", response_model=AudioTranscribeResponse)
async def transcribe_audio_file(
    file: UploadFile = File(...),
    model: str = "whisper-base",
    language: Optional[str] = None,
    task: str = "transcribe",
    word_timestamps: bool = False,
    api_key: str = Depends(shared.verify_api_key)
):
    """
    Transcribe audio file using Whisper (faster-whisper).

    **Models:**
    - whisper-tiny: Fastest, ~40MB, 7.8% WER
    - whisper-base: Balanced (default), ~1GB RAM, 5.0% WER
    - whisper-small: Better accuracy, ~2GB RAM, 3.4% WER
    - whisper-medium: High accuracy, ~5GB RAM, 2.9% WER

    **Supported formats:** mp3, wav, m4a, ogg, flac, webm

    **Usage:**
    ```bash
    curl -X POST http://localhost:11435/api/transcribe \\
      -H "X-API-Key: YOUR_API_KEY" \\
      -F "file=@audio.mp3" \\
      -F "model=whisper-base" \\
      -F "language=en"
    ```

    **Returns:**
    - text: Full transcribed text
    - language: Detected/specified language
    - language_probability: Confidence of language detection
    - duration: Audio duration in seconds
    - segments: Timestamped segments (with optional word timestamps)
    - latency_ms: Processing time
    """

    if not shared.WHISPER_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Whisper not available. Install with: pip install faster-whisper"
        )

    # Validate file type
    valid_types = {
        'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/x-wav',
        'audio/ogg', 'audio/flac', 'audio/x-flac', 'audio/x-m4a',
        'audio/mp4', 'audio/webm', 'video/webm'
    }
    if file.content_type and not any(
        file.content_type.startswith(t.split('/')[0]) for t in valid_types
    ):
        logger.warning(f"Unexpected content type: {file.content_type}, attempting anyway...")

    # Parse model size from model name
    model_size = model.replace("whisper-", "") if model.startswith("whisper-") else model
    if model_size not in ["tiny", "base", "small", "medium", "large-v3"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model}. Available: whisper-tiny, whisper-base, whisper-small, whisper-medium"
        )

    try:
        # Read audio file
        audio_bytes = await file.read()

        # Get whisper handler
        handler = shared.get_whisper_handler()

        # Transcribe
        start_time = time.perf_counter()
        result = handler.transcribe(
            audio=audio_bytes,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            model_size=model_size,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Transcribed {result.duration:.1f}s audio with whisper-{model_size} "
            f"in {latency_ms:.0f}ms (language: {result.language})"
        )

        return AudioTranscribeResponse(
            model=f"whisper-{model_size}",
            text=result.text,
            language=result.language,
            language_probability=result.language_probability,
            duration=result.duration,
            segments=[s.to_dict() for s in result.segments],
            latency_ms=round(latency_ms, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/transcribe/base64", response_model=AudioTranscribeResponse)
async def transcribe_audio_base64(
    request: AudioTranscribeRequest,
    api_key: str = Depends(shared.verify_api_key)
):
    """
    Transcribe audio from base64 encoded data.

    **Usage:**
    ```bash
    curl -X POST http://localhost:11435/api/transcribe/base64 \\
      -H "X-API-Key: YOUR_API_KEY" \\
      -H "Content-Type: application/json" \\
      -d '{
        "audio": "data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAA...",
        "model": "whisper-base",
        "language": "en"
      }'
    ```

    **Returns:**
    - text: Full transcribed text
    - language: Detected/specified language
    - segments: Timestamped segments
    - latency_ms: Processing time
    """

    if not shared.WHISPER_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Whisper not available. Install with: pip install faster-whisper"
        )

    # Parse model size
    model_size = request.model.replace("whisper-", "") if request.model.startswith("whisper-") else request.model
    if model_size not in ["tiny", "base", "small", "medium", "large-v3"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {request.model}. Available: whisper-tiny, whisper-base, whisper-small, whisper-medium"
        )

    try:
        # Decode base64 audio
        audio_data = request.audio
        if audio_data.startswith("data:"):
            # Remove data URI prefix
            audio_data = audio_data.split(",", 1)[1]

        audio_bytes = base64.b64decode(audio_data)

        # Get whisper handler
        handler = shared.get_whisper_handler()

        # Transcribe
        start_time = time.perf_counter()
        result = handler.transcribe(
            audio=audio_bytes,
            language=request.language,
            task=request.task,
            word_timestamps=request.word_timestamps,
            model_size=model_size,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Transcribed {result.duration:.1f}s audio (base64) with whisper-{model_size} "
            f"in {latency_ms:.0f}ms"
        )

        return AudioTranscribeResponse(
            model=f"whisper-{model_size}",
            text=result.text,
            language=result.language,
            language_probability=result.language_probability,
            duration=result.duration,
            segments=[s.to_dict() for s in result.segments],
            latency_ms=round(latency_ms, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/audio/embed", response_model=AudioEmbedResponse)
async def embed_audio(
    file: UploadFile = File(...),
    whisper_model: str = "whisper-base",
    embedding_model: str = "m2v-bge-m3-1024d",
    language: Optional[str] = None,
    api_key: str = Depends(shared.verify_api_key)
):
    """
    Transcribe audio and generate embeddings for indexation.

    Pipeline: Audio -> Whisper Transcription -> Text Embedding

    **Use case:** Index audio/video content for semantic search.

    **Usage:**
    ```bash
    curl -X POST http://localhost:11435/api/audio/embed \\
      -H "X-API-Key: YOUR_API_KEY" \\
      -F "file=@podcast.mp3" \\
      -F "whisper_model=whisper-base" \\
      -F "embedding_model=m2v-bge-m3-1024d"
    ```

    **Returns:**
    - text: Transcribed text
    - language: Detected language
    - embeddings: Vector embeddings of transcribed text
    - latency_ms: Total processing time
    """

    if not shared.WHISPER_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Whisper not available. Install with: pip install faster-whisper"
        )

    # Validate embedding model
    if embedding_model not in shared.model_manager.configs:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding model '{embedding_model}' not found"
        )

    # Parse whisper model size
    model_size = whisper_model.replace("whisper-", "") if whisper_model.startswith("whisper-") else whisper_model
    if model_size not in ["tiny", "base", "small", "medium", "large-v3"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid whisper model: {whisper_model}"
        )

    try:
        start_time = time.perf_counter()

        # Step 1: Transcribe audio
        audio_bytes = await file.read()
        handler = shared.get_whisper_handler()

        transcription = handler.transcribe(
            audio=audio_bytes,
            language=language,
            model_size=model_size,
        )

        # Step 2: Generate embeddings from transcribed text
        embed_model = shared.model_manager.get_model(embedding_model)
        embeddings = embed_model.encode([transcription.text], show_progress_bar=False)

        # Handle 2D Matryoshka truncation
        truncate_dims = getattr(embed_model, '_truncate_dims', None)
        if truncate_dims:
            embeddings = embeddings[:, :truncate_dims]

        embeddings_list = [emb.tolist() for emb in embeddings]

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Audio embedded: {transcription.duration:.1f}s audio -> "
            f"{len(transcription.text)} chars -> {len(embeddings_list[0])}D embedding "
            f"in {latency_ms:.0f}ms"
        )

        return AudioEmbedResponse(
            model=f"whisper-{model_size}",
            text=transcription.text,
            language=transcription.language,
            embedding_model=embedding_model,
            embeddings=embeddings_list,
            latency_ms=round(latency_ms, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
