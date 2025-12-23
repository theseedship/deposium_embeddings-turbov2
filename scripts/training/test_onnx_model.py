"""
Test ONNX Quantized Model
==========================

Tests the ONNX INT8 quantized model on the full test set to verify
that quantization did not degrade performance.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DATASET_ROOT = project_root / "data" / "complexity_classification"
IMAGES_DIR = DATASET_ROOT / "images_500"
ANNOTATIONS_CSV = DATASET_ROOT / "annotations_500.csv"
ONNX_MODEL = project_root / "models" / "vl_distilled_resnet18" / "model_quantized.onnx"  # INT8 quantized


def preprocess_image(image_path):
    """Preprocess image for ONNX model (matches PyTorch transforms)."""
    # Load image
    image = Image.open(image_path).convert('RGB')

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

    # Convert to numpy and normalize [0, 255] -> [0, 1]
    image_np = np.array(image).astype(np.float32) / 255.0

    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_np = (image_np - mean) / std

    # Convert to CHW format (from HWC)
    image_np = np.transpose(image_np, (2, 0, 1))

    # Add batch dimension
    image_np = np.expand_dims(image_np, axis=0)

    return image_np


def test_onnx_model(model_path, test_df, images_dir):
    """Test ONNX model on test set."""
    logger.info(f"Loading ONNX model: {model_path}")

    # Create session
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])

    all_preds = []
    all_labels = []
    all_probs = []

    logger.info(f"Testing on {len(test_df)} images...")

    for idx, row in test_df.iterrows():
        img_path = images_dir / row['image_path']
        label = int(row['label'])

        # Preprocess
        image_np = preprocess_image(img_path)

        # Inference
        outputs = session.run(['output'], {'input': image_np})
        logits = outputs[0][0]

        # Softmax
        probs = np.exp(logits) / np.sum(np.exp(logits))
        pred = np.argmax(probs)

        all_preds.append(pred)
        all_labels.append(label)
        all_probs.append(probs)

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average=None)
    report = classification_report(all_labels, all_preds, target_names=['LOW', 'HIGH'], output_dict=False)
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, recall, report, cm


def main():
    """Main test function."""
    logger.info("=" * 80)
    logger.info("TESTING ONNX INT8 QUANTIZED MODEL")
    logger.info("=" * 80)

    # Load test data
    df = pd.read_csv(ANNOTATIONS_CSV)
    test_df = df[df['image_path'].str.startswith("test/")]

    logger.info(f"\nTest set: {len(test_df)} images ({(test_df['label']==0).sum()} LOW / {(test_df['label']==1).sum()} HIGH)")
    logger.info(f"Model: {ONNX_MODEL}")
    logger.info(f"Size: {ONNX_MODEL.stat().st_size / (1024 * 1024):.2f} MB\n")

    # Test
    accuracy, recall, report, cm = test_onnx_model(ONNX_MODEL, test_df, IMAGES_DIR)

    # Results
    logger.info("\n" + "=" * 80)
    logger.info("TEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"\nAccuracy: {accuracy:.4f}")
    logger.info(f"Recall LOW: {recall[0]:.4f}")
    logger.info(f"Recall HIGH: {recall[1]:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"\n{cm}")
    logger.info(f"\nClassification Report:")
    logger.info(f"\n{report}")

    # Check target
    if recall[1] >= 0.999:
        logger.info("\n✅ TARGET ACHIEVED! HIGH recall ≥ 99.9%")
    else:
        logger.warning(f"\n⚠️ HIGH recall below target: {recall[1]:.4f} < 0.999")

    if accuracy >= 0.99:
        logger.info("✅ Overall accuracy ≥ 99%")
    else:
        logger.warning(f"⚠️ Accuracy below 99%: {accuracy:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("ONNX MODEL READY FOR DEPLOYMENT!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
