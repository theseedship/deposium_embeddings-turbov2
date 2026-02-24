"""Document classification routes."""
import base64
import gc
import io
import logging
import time
import torch
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image

from .. import shared
from ..classifier import get_classifier, ClassifyRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/classify/file")
async def classify_document_file(
    file: UploadFile = File(...),
    model: str = "vl-classifier",
    api_key: str = Depends(shared.verify_api_key)
):
    """
    Classify document complexity from uploaded file (multipart/form-data).

    **Models:**
    - vl-classifier: ResNet18 ONNX (fast, ~10ms)
    - lfm25-vl: LFM2.5-VL-1.6B VLM (accurate, ~10-15s)

    **Usage:**
    ```bash
    curl -X POST http://localhost:11435/api/classify/file \\
      -H "X-API-Key: YOUR_API_KEY" \\
      -F "file=@document.jpg" \\
      -F "model=vl-classifier"
    ```

    **Returns:**
    - class_name: "LOW" (simple OCR) or "HIGH" (VLM reasoning)
    - confidence: 0.0-1.0
    - probabilities: {"LOW": float, "HIGH": float}
    - routing_decision: str (routing recommendation)
    - latency_ms: float

    **Use case:** Route LOW complexity to OCR (~100ms), HIGH to VLM (~2000ms)
    """

    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Expected image/*"
            )

        # Use LFM2.5-VL for classification if requested
        if model == "lfm25-vl":
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            vlm_model, vlm_processor = shared.model_manager.get_model("lfm25-vl")

            # Classification prompt
            prompt = """Analyze this document image and classify its complexity:
- LOW: Simple text, single column, no tables, easy to OCR
- HIGH: Complex layout, tables, forms, multiple columns, needs VLM reasoning

Answer with ONLY one word: LOW or HIGH"""

            conversation = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}]

            inputs = vlm_processor.apply_chat_template(
                conversation, add_generation_prompt=True,
                return_tensors="pt", return_dict=True, tokenize=True
            ).to(vlm_model.device)

            start_time = time.time()
            with torch.inference_mode():
                outputs = vlm_model.generate(**inputs, max_new_tokens=10)
            latency_ms = (time.time() - start_time) * 1000

            response = vlm_processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # Free CUDA tensors immediately
            del inputs, outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Parse response
            is_high = "HIGH" in response.upper()
            class_name = "HIGH" if is_high else "LOW"

            return {
                "class_name": class_name,
                "confidence": 0.85,  # VLM doesn't provide exact confidence
                "probabilities": {"LOW": 0.15 if is_high else 0.85, "HIGH": 0.85 if is_high else 0.15},
                "routing_decision": f"Route to {'VLM reasoning' if is_high else 'fast OCR'}",
                "latency_ms": round(latency_ms, 2),
                "model": "lfm25-vl"
            }

        # Default: Use ResNet18 classifier
        classifier = get_classifier()
        result = await classifier.predict_from_file(file)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/classify/base64")
async def classify_document_base64(request: ClassifyRequest, api_key: str = Depends(shared.verify_api_key)):
    """
    Classify document complexity from base64 encoded image (application/json).

    **Models:**
    - vl-classifier: ResNet18 ONNX (fast, ~10ms) - default
    - lfm25-vl: LFM2.5-VL-1.6B VLM (accurate, ~10-15s)

    **Usage:**
    ```bash
    curl -X POST http://localhost:11435/api/classify/base64 \\
      -H "X-API-Key: YOUR_API_KEY" \\
      -H "Content-Type: application/json" \\
      -d '{"image":"data:image/jpeg;base64,/9j/4AAQ...", "model":"lfm25-vl"}'
    ```

    **Returns:**
    - class_name: "LOW" (simple OCR) or "HIGH" (VLM reasoning)
    - confidence: 0.0-1.0
    - probabilities: {"LOW": float, "HIGH": float}
    - routing_decision: str (routing recommendation)
    - latency_ms: float

    **Use case:** Route LOW complexity to OCR (~100ms), HIGH to VLM (~2000ms)
    """

    try:
        # Validate input
        if request.image is None:
            raise HTTPException(
                status_code=400,
                detail="Missing 'image' field in JSON body"
            )

        # Use LFM2.5-VL for classification if requested
        if request.model == "lfm25-vl":
            # Decode base64 image
            image_data = request.image
            if image_data.startswith("data:"):
                image_data = image_data.split(",", 1)[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            vlm_model, vlm_processor = shared.model_manager.get_model("lfm25-vl")

            # Classification prompt
            prompt = """Analyze this document image and classify its complexity:
- LOW: Simple text, single column, no tables, easy to OCR
- HIGH: Complex layout, tables, forms, multiple columns, needs VLM reasoning

Answer with ONLY one word: LOW or HIGH"""

            conversation = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}]

            inputs = vlm_processor.apply_chat_template(
                conversation, add_generation_prompt=True,
                return_tensors="pt", return_dict=True, tokenize=True
            ).to(vlm_model.device)

            start_time = time.time()
            with torch.inference_mode():
                outputs = vlm_model.generate(**inputs, max_new_tokens=10)
            latency_ms = (time.time() - start_time) * 1000

            response = vlm_processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # Free CUDA tensors immediately
            del inputs, outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Parse response
            is_high = "HIGH" in response.upper()
            class_name = "HIGH" if is_high else "LOW"

            return {
                "class_name": class_name,
                "confidence": 0.85,  # VLM doesn't provide exact confidence
                "probabilities": {"LOW": 0.15 if is_high else 0.85, "HIGH": 0.85 if is_high else 0.15},
                "routing_decision": f"Route to {'VLM reasoning' if is_high else 'fast OCR'}",
                "latency_ms": round(latency_ms, 2),
                "model": "lfm25-vl"
            }

        # Default: Use ResNet18 classifier
        classifier = get_classifier()
        result = await classifier.predict_from_base64(request.image)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
