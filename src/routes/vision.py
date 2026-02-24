"""Vision-language routes."""
import base64
import gc
import io
import logging
import time
import torch
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image

from .. import shared
from ..schemas.requests import VisionRequest, VisionResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/vision", response_model=VisionResponse)
async def process_vision(request: VisionRequest, api_key: str = Depends(shared.verify_api_key)):
    """
    Process an image with a vision-language model for OCR and document understanding.

    **Models:**
    - lfm25-vl: LFM2.5-VL-1.6B (excellent OCR, edge-first design)

    **Usage:**
    ```bash
    curl -X POST http://localhost:11436/api/vision \\
      -H "X-API-Key: YOUR_API_KEY" \\
      -H "Content-Type: application/json" \\
      -d '{
        "model": "lfm25-vl",
        "image": "data:image/png;base64,iVBORw0KGgo...",
        "prompt": "Extract all text from this document."
      }'
    ```

    **Prompts examples:**
    - "Extract all text from this document." (OCR)
    - "Is this document SIMPLE or COMPLEX? Answer with just one word." (Classification)
    - "Summarize the main content in 2-3 sentences." (Summary)
    - "What type of document is this?" (Document type detection)

    **Returns:**
    - model: Model used
    - response: Generated text response
    - latency_ms: Processing time in milliseconds
    """

    # Validate model
    if request.model not in shared.model_manager.configs:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found. Available vision models: lfm25-vl"
        )

    model_config = shared.model_manager.configs[request.model]
    if model_config.type != "vision_language":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is not a vision-language model"
        )

    try:
        # Decode base64 image
        image_data = request.image
        if image_data.startswith("data:"):
            # Remove data URI prefix (e.g., "data:image/png;base64,")
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Get model (lazy loading) - returns (model, processor) tuple
        vlm_model, vlm_processor = shared.model_manager.get_model(request.model)

        # Prepare conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": request.prompt},
                ],
            },
        ]

        # Process inputs
        inputs = vlm_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(vlm_model.device)

        # Generate response (no_grad prevents computation graph retention)
        start_time = time.time()
        with torch.inference_mode():
            outputs = vlm_model.generate(**inputs, max_new_tokens=request.max_tokens or 512)
        latency_ms = (time.time() - start_time) * 1000

        # Decode response
        response_text = vlm_processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Free CUDA tensors immediately
        del inputs, outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Extract just the assistant's response (remove the prompt echo)
        if "assistant" in response_text.lower():
            # Try to extract text after "assistant" marker
            parts = response_text.split("assistant")
            if len(parts) > 1:
                response_text = parts[-1].strip()

        logger.info(f"Vision processed with {request.model} in {latency_ms:.0f}ms")

        return VisionResponse(
            model=request.model,
            response=response_text,
            latency_ms=round(latency_ms, 2)
        )

    except Exception as e:
        logger.error(f"Vision processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/vision/file")
async def process_vision_file(
    file: UploadFile = File(...),
    prompt: str = "Extract all text from this document.",
    model: str = "lfm25-vl",
    max_tokens: int = 512,
    api_key: str = Depends(shared.verify_api_key)
):
    """
    Process an uploaded image file with a vision-language model.

    **Usage:**
    ```bash
    curl -X POST http://localhost:11436/api/vision/file \\
      -H "X-API-Key: YOUR_API_KEY" \\
      -F "file=@document.png" \\
      -F "prompt=Extract all text from this document."
    ```

    **Returns:**
    - model: Model used
    - response: Generated text response
    - latency_ms: Processing time in milliseconds
    """

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected image/*"
        )

    # Validate model
    if model not in shared.model_manager.configs:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' not found. Available vision models: lfm25-vl"
        )

    model_config = shared.model_manager.configs[model]
    if model_config.type != "vision_language":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not a vision-language model"
        )

    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Get model (lazy loading) - returns (model, processor) tuple
        vlm_model, vlm_processor = shared.model_manager.get_model(model)

        # Prepare conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Process inputs
        inputs = vlm_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(vlm_model.device)

        # Generate response (no_grad prevents computation graph retention)
        start_time = time.time()
        with torch.inference_mode():
            outputs = vlm_model.generate(**inputs, max_new_tokens=max_tokens)
        latency_ms = (time.time() - start_time) * 1000

        # Decode response
        response_text = vlm_processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Free CUDA tensors immediately
        del inputs, outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Extract just the assistant's response
        if "assistant" in response_text.lower():
            parts = response_text.split("assistant")
            if len(parts) > 1:
                response_text = parts[-1].strip()

        logger.info(f"Vision file processed with {model} in {latency_ms:.0f}ms")

        return {
            "model": model,
            "response": response_text,
            "latency_ms": round(latency_ms, 2)
        }

    except Exception as e:
        logger.error(f"Vision file processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
