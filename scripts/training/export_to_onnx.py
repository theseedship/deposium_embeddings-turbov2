"""
Export Distilled ResNet18 to ONNX INT8
=======================================

Exports the trained ResNet18 student model to ONNX format with INT8 quantization.
Target: ~11MB ONNX model
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torchvision import models
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
MODEL_PATH = project_root / "models" / "vl_distilled_resnet18" / "best_student.pth"
ONNX_PATH = project_root / "models" / "vl_distilled_resnet18" / "model.onnx"
ONNX_QUANTIZED_PATH = project_root / "models" / "vl_distilled_resnet18" / "model_quantized.onnx"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNet18Student(nn.Module):
    """ResNet18 student model."""

    def __init__(self, pretrained=False, feature_dim=768):
        super().__init__()

        # Load ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)

        # Get feature dimension (before final FC layer)
        self.resnet_feature_dim = self.resnet.fc.in_features  # 512

        # Replace final FC with custom layers
        self.resnet.fc = nn.Identity()  # Remove original FC

        # Feature projection (to match CLIP feature_dim)
        self.feature_projection = nn.Sequential(
            nn.Linear(self.resnet_feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Classifier
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, x):
        """Extract features and logits."""
        # ResNet18 backbone
        x = self.resnet(x)  # (batch, 512)

        # Project to CLIP feature space
        features = self.feature_projection(x)  # (batch, feature_dim)

        # Classifier
        logits = self.classifier(features)

        return logits  # Only return logits for inference


def load_model(model_path, device):
    """Load trained model."""
    logger.info(f"Loading model from {model_path}")

    # Initialize model
    model = ResNet18Student(pretrained=False, feature_dim=768)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['student_state_dict'])

    model.eval()
    logger.info(f"Model loaded (epoch {checkpoint['epoch']}, val HIGH recall: {checkpoint['val_recall_high']:.4f})")

    return model


def export_to_onnx(model, onnx_path):
    """Export model to ONNX."""
    logger.info(f"\nExporting to ONNX: {onnx_path}")

    # Dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    logger.info(f"✅ ONNX export complete: {onnx_path}")

    # Verify
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("✅ ONNX model verified")

    # File size
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX model size: {size_mb:.2f} MB")

    return onnx_model


def quantize_onnx(onnx_path, quantized_path):
    """Quantize ONNX model to INT8."""
    logger.info(f"\nQuantizing to INT8: {quantized_path}")

    quantize_dynamic(
        str(onnx_path),
        str(quantized_path),
        weight_type=QuantType.QUInt8
    )

    logger.info(f"✅ Quantization complete: {quantized_path}")

    # File size
    size_mb = quantized_path.stat().st_size / (1024 * 1024)
    logger.info(f"Quantized model size: {size_mb:.2f} MB")

    return quantized_path


def test_onnx_inference(onnx_path):
    """Test ONNX model inference."""
    import onnxruntime as ort
    import numpy as np

    logger.info(f"\nTesting ONNX inference: {onnx_path}")

    # Load session
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    # Dummy input
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Run inference
    outputs = session.run(['output'], {'input': dummy_input})
    logits = outputs[0]

    logger.info(f"Input shape: {dummy_input.shape}")
    logger.info(f"Output shape: {logits.shape}")
    logger.info(f"Output logits: {logits[0]}")

    # Apply softmax
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    predicted_class = np.argmax(probs, axis=1)[0]

    logger.info(f"Probabilities: {probs[0]}")
    logger.info(f"Predicted class: {predicted_class} ({'HIGH' if predicted_class == 1 else 'LOW'})")

    logger.info("✅ ONNX inference successful")


def main():
    """Main export function."""
    logger.info("=" * 80)
    logger.info("EXPORT DISTILLED RESNET18 TO ONNX INT8")
    logger.info("=" * 80)

    # Load PyTorch model
    model = load_model(MODEL_PATH, DEVICE)

    # Export to ONNX
    onnx_model = export_to_onnx(model, ONNX_PATH)

    # Quantize to INT8
    quantized_path = quantize_onnx(ONNX_PATH, ONNX_QUANTIZED_PATH)

    # Test inference
    test_onnx_inference(ONNX_QUANTIZED_PATH)

    logger.info("\n" + "=" * 80)
    logger.info("EXPORT COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nFinal model: {ONNX_QUANTIZED_PATH}")
    logger.info(f"Size: {ONNX_QUANTIZED_PATH.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
