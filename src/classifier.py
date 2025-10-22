"""
Document Complexity Classifier Module
======================================

Binary classifier for routing documents to OCR or VLM pipelines.
- LOW complexity: Plain text documents → OCR (~100ms)
- HIGH complexity: Charts/diagrams → VLM reasoning (~2000ms)

Model: ResNet18 ONNX quantized (11MB, ~10ms inference)
Accuracy: 93% overall, 100% HIGH recall
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


class ComplexityClassifier:
    """
    Document complexity classifier with lazy loading.

    Loads ONNX model only on first prediction to save memory.
    """

    def __init__(self, model_path: str = "src/models/complexity_classifier/model_quantized.onnx"):
        """
        Initialize classifier (lazy loading).

        Args:
            model_path: Path to ONNX quantized model
        """
        self.model_path = Path(model_path)
        self.session = None
        self.class_names = ["LOW", "HIGH"]

        # Image preprocessing parameters (ImageNet normalization)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        logger.info(f"Complexity classifier initialized (lazy loading from {model_path})")

    def _load_model(self):
        """Load ONNX model (called on first prediction)."""
        if self.session is not None:
            return  # Already loaded

        import onnxruntime as ort

        logger.info("=" * 80)
        logger.info("🎯 Loading Complexity Classifier (ONNX INT8)")
        logger.info("=" * 80)
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  Size: {self.model_path.stat().st_size / 1024 / 1024:.1f} MB")
        logger.info("  Accuracy: 93% overall, 100% HIGH recall")
        logger.info("  Latency: ~10ms on CPU")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load ONNX session (CPU optimized)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider']
        )

        logger.info("✅ Complexity classifier loaded!")
        logger.info("=" * 80)

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for inference.

        Args:
            image: PIL Image

        Returns:
            Numpy array [1, 3, 224, 224] ready for ONNX
        """
        # Resize to 256x256, then center crop to 224x224
        image = image.convert('RGB')
        image = image.resize((256, 256), Image.Resampling.BILINEAR)

        # Center crop
        left = (256 - 224) // 2
        top = (256 - 224) // 2
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
        # Lazy load model on first prediction
        if self.session is None:
            self._load_model()

        start_time = time.perf_counter()

        # Preprocess image
        img_array = self._preprocess_image(image)

        # Run ONNX inference
        outputs = self.session.run(None, {'input': img_array})[0]

        # Softmax to get probabilities
        exp_outputs = np.exp(outputs[0] - np.max(outputs[0]))
        probs = exp_outputs / np.sum(exp_outputs)

        # Get prediction
        class_id = int(np.argmax(probs))
        class_name = self.class_names[class_id]
        confidence = float(probs[class_id])

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Routing decision
        if class_name == "LOW":
            routing = "Simple document - Route to OCR pipeline (~100ms)"
        else:
            routing = "Complex document - Route to VLM reasoning pipeline (~2000ms)"

        # Log prediction
        logger.info(f"Classified: {class_name} ({confidence*100:.1f}%) - {latency_ms:.1f}ms")

        return {
            'class_name': class_name,
            'class_id': class_id,
            'confidence': confidence,
            'probabilities': {
                'LOW': float(probs[0]),
                'HIGH': float(probs[1])
            },
            'routing_decision': routing,
            'latency_ms': latency_ms
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
