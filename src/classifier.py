"""
Document Complexity Classifier Module
======================================

Binary classifier for routing documents to OCR or VLM pipelines.
- LOW complexity: Plain text documents → OCR (~100ms)
- HIGH complexity: Charts/diagrams → VLM reasoning (~2000ms)

Primary model: CLIP ViT-B/32 ONNX uint8 (153MB, ~20ms/image, zero-shot)
  - Labels: "SIMPLE" → LOW, "COMPLEX" → HIGH
  - Text encoder runs once at startup, vision encoder per image
Fallback:  ResNet18 ONNX quantized (11MB, ~10ms, distilled from CLIP)
"""

from fastapi import UploadFile
from pydantic import BaseModel
from typing import Optional, Dict
from PIL import Image
import numpy as np
import logging
import time
import base64
import io
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)


class ClassifyRequest(BaseModel):
    """Request model for complexity classification."""
    image: Optional[str] = None  # Base64 encoded image
    model: str = "clip-classifier"  # Model to use: "clip-classifier" (CLIP ONNX), "vl-classifier" (ResNet18), "lfm25-vl" (VLM)


class ComplexityClassifier:
    """
    Document complexity classifier with lazy loading.

    Loads ONNX model only on first prediction to save memory.
    """

    def __init__(self):
        """
        Initialize classifier.
        """
        self.class_names = ["LOW", "HIGH"]

        # Image preprocessing parameters (ImageNet normalization)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        logger.info("Complexity classifier initialized (delegating to ModelManager)")

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for inference (matches PyTorch training transforms).

        Args:
            image: PIL Image

        Returns:
            Numpy array [1, 3, 224, 224] ready for ONNX
        """
        # Convert to RGB
        image = image.convert('RGB')

        # Resize shortest side to 256 (maintaining aspect ratio)
        w, h = image.size
        if w < h:
            new_w, new_h = 256, int(256 * h / w)
        else:
            new_h, new_w = 256, int(256 * w / h)
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Center crop to 224x224
        left = (new_w - 224) // 2
        top = (new_h - 224) // 2
        image = image.crop((left, top, left + 224, top + 224))

        # Convert to numpy array [224, 224, 3]
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Normalize (ImageNet stats)
        img_array = (img_array - self.mean) / self.std

        # Transpose to [3, 224, 224] and add batch dimension [1, 3, 224, 224]
        img_array = img_array.transpose(2, 0, 1)[np.newaxis, :]

        return img_array.astype(np.float32)

    def _decode_base64_image(self, base64_str: str) -> Image.Image:
        """
        Decode base64 string to PIL Image.

        Args:
            base64_str: Base64 encoded image (with or without data URI prefix)

        Returns:
            PIL Image
        """
        # Remove data URI prefix if present (data:image/jpeg;base64,...)
        if ',' in base64_str:
            base64_str = base64_str.split(',', 1)[1]

        # Decode base64
        image_bytes = base64.b64decode(base64_str)

        # Open as PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        return image

    async def predict_from_base64(self, base64_image: str) -> Dict:
        """
        Predict from base64 encoded image.

        Args:
            base64_image: Base64 encoded image string

        Returns:
            Prediction dictionary
        """
        image = self._decode_base64_image(base64_image)
        return self.predict(image)

    async def predict_from_file(self, file: UploadFile) -> Dict:
        """
        Predict from uploaded file.

        Args:
            file: FastAPI UploadFile

        Returns:
            Prediction dictionary
        """
        # Read file bytes
        image_bytes = await file.read()

        # Open as PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        return self.predict(image)

    def predict(self, image: Image.Image) -> Dict:
        """
        Run complexity classification on image.

        Tries CLIP zero-shot classifier first (clip-classifier), falls back to
        ResNet18 ONNX (vl-classifier) if CLIP is unavailable.

        Args:
            image: PIL Image

        Returns:
            {
                'class_name': 'LOW' or 'HIGH',
                'class_id': 0 or 1,
                'confidence': float,
                'probabilities': {'LOW': float, 'HIGH': float},
                'routing_decision': str,
                'latency_ms': float
            }
        """
        from src.model_manager import get_model_manager
        manager = get_model_manager()

        # ── CLIP zero-shot (primary) ──────────────────────────────────────────
        if "clip-classifier" in manager.configs:
            try:
                clip_model = manager.get_model("clip-classifier")
                result = clip_model.predict(image)

                # Map CLIP labels to routing labels: SIMPLE→LOW, COMPLEX→HIGH
                clip_name = result["class_name"]
                class_name = "LOW" if clip_name == "SIMPLE" else "HIGH"
                class_id   = self.class_names.index(class_name)
                confidence  = result["confidence"]
                clip_probs  = result["probabilities"]

                routing = (
                    "Simple document - Route to OCR pipeline (~100ms)"
                    if class_name == "LOW"
                    else "Complex document - Route to VLM reasoning pipeline (~2000ms)"
                )

                logger.info(
                    f"CLIP classified: {clip_name}→{class_name} "
                    f"({confidence*100:.1f}%) - {result['latency_ms']:.1f}ms"
                )

                return {
                    "class_name":  class_name,
                    "class_id":    class_id,
                    "confidence":  confidence,
                    "probabilities": {
                        "LOW":  clip_probs.get("SIMPLE", 0.0),
                        "HIGH": clip_probs.get("COMPLEX", 0.0),
                    },
                    "routing_decision": routing,
                    "latency_ms": result["latency_ms"],
                }
            except Exception as e:
                logger.warning(f"CLIP classifier failed, falling back to ResNet18: {e}")

        # ── ResNet18 ONNX fallback ────────────────────────────────────────────
        session = manager.get_model("vl-classifier")

        start_time = time.perf_counter()
        img_array  = self._preprocess_image(image)
        outputs    = session.run(None, {"input": img_array})[0]

        exp_outputs = np.exp(outputs[0] - np.max(outputs[0]))
        probs       = exp_outputs / np.sum(exp_outputs)

        class_id   = int(np.argmax(probs))
        class_name = self.class_names[class_id]
        confidence = float(probs[class_id])
        latency_ms = (time.perf_counter() - start_time) * 1000

        routing = (
            "Simple document - Route to OCR pipeline (~100ms)"
            if class_name == "LOW"
            else "Complex document - Route to VLM reasoning pipeline (~2000ms)"
        )

        logger.info(f"ResNet18 classified: {class_name} ({confidence*100:.1f}%) - {latency_ms:.1f}ms")

        return {
            "class_name":  class_name,
            "class_id":    class_id,
            "confidence":  confidence,
            "probabilities": {
                "LOW":  float(probs[0]),
                "HIGH": float(probs[1]),
            },
            "routing_decision": routing,
            "latency_ms": latency_ms,
        }


# Global classifier instance (lazy loading)
_classifier_instance = None


def get_classifier() -> ComplexityClassifier:
    """
    Get global classifier instance (singleton with lazy loading).

    Returns:
        ComplexityClassifier instance
    """
    global _classifier_instance

    if _classifier_instance is None:
        _classifier_instance = ComplexityClassifier()

    return _classifier_instance
