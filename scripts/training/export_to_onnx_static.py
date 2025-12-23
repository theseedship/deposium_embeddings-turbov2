"""
Export to ONNX with Static Quantization
========================================

Uses static quantization with calibration data for better accuracy.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torchvision import models
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import onnxruntime as ort
import numpy as np
from PIL import Image
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
MODEL_PATH = project_root / "models" / "vl_distilled_resnet18" / "best_student.pth"
ONNX_PATH = project_root / "models" / "vl_distilled_resnet18" / "model.onnx"
ONNX_QUANTIZED_PATH = project_root / "models" / "vl_distilled_resnet18" / "model_quantized_static.onnx"
DATASET_ROOT = project_root / "data" / "complexity_classification"
IMAGES_DIR = DATASET_ROOT / "images_500"
ANNOTATIONS_CSV = DATASET_ROOT / "annotations_500.csv"


class ResNet18Student(nn.Module):
    """ResNet18 student model."""

    def __init__(self, pretrained=False, feature_dim=768):
        super().__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet_feature_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.feature_projection = nn.Sequential(
            nn.Linear(self.resnet_feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, x):
        x = self.resnet(x)
        features = self.feature_projection(x)
        logits = self.classifier(features)
        return logits


class CalibrationReader(CalibrationDataReader):
    """Calibration data reader for static quantization."""

    def __init__(self, images_dir, annotations_csv, num_samples=50):
        self.images_dir = Path(images_dir)

        # Load training data for calibration
        df = pd.read_csv(annotations_csv)
        train_df = df[df['image_path'].str.startswith("train/")]

        # Sample calibration images
        self.calibration_images = train_df.sample(n=min(num_samples, len(train_df)), random_state=42)
        self.current_idx = 0

        logger.info(f"Calibration data: {len(self.calibration_images)} images")

    def get_next(self):
        """Get next calibration sample."""
        if self.current_idx >= len(self.calibration_images):
            return None

        row = self.calibration_images.iloc[self.current_idx]
        img_path = self.images_dir / row['image_path']

        # Preprocess (matches PyTorch transforms)
        image = Image.open(img_path).convert('RGB')

        # Resize shortest side to 256 (maintaining aspect ratio)
        w, h = image.size
        if w < h:
            new_w = 256
            new_h = int(256 * h / w)
        else:
            new_h = 256
            new_w = int(256 * w / h)
        image = image.resize((new_w, new_h), Image.BILINEAR)

        # Center crop to 224x224
        left = (new_w - 224) // 2
        top = (new_h - 224) // 2
        image = image.crop((left, top, left + 224, top + 224))

        image_np = np.array(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_np = (image_np - mean) / std
        image_np = np.transpose(image_np, (2, 0, 1))
        image_np = np.expand_dims(image_np, axis=0)

        self.current_idx += 1

        return {'input': image_np}


def load_and_export_onnx(model_path, onnx_path):
    """Load PyTorch model and export to ONNX."""
    logger.info(f"Loading model from {model_path}")

    model = ResNet18Student(pretrained=False, feature_dim=768)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['student_state_dict'])
    model.eval()

    logger.info(f"Exporting to ONNX: {onnx_path}")

    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

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

    logger.info(f"✅ ONNX export complete")
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX model size: {size_mb:.2f} MB")


def quantize_static_onnx(onnx_path, quantized_path, calibration_reader):
    """Quantize ONNX model with static quantization."""
    logger.info(f"\nStatic quantization to INT8: {quantized_path}")

    quantize_static(
        str(onnx_path),
        str(quantized_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8
    )

    logger.info(f"✅ Static quantization complete")
    size_mb = quantized_path.stat().st_size / (1024 * 1024)
    logger.info(f"Quantized model size: {size_mb:.2f} MB")


def main():
    """Main export function."""
    logger.info("=" * 80)
    logger.info("EXPORT TO ONNX WITH STATIC QUANTIZATION")
    logger.info("=" * 80)

    # Export to ONNX
    if not ONNX_PATH.exists():
        load_and_export_onnx(MODEL_PATH, ONNX_PATH)

    # Create calibration reader
    calibration_reader = CalibrationReader(IMAGES_DIR, ANNOTATIONS_CSV, num_samples=100)

    # Static quantization
    quantize_static_onnx(ONNX_PATH, ONNX_QUANTIZED_PATH, calibration_reader)

    logger.info("\n" + "=" * 80)
    logger.info("EXPORT COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nQuantized model: {ONNX_QUANTIZED_PATH}")
    logger.info(f"Size: {ONNX_QUANTIZED_PATH.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
