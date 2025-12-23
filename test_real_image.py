"""
Test real image with ONNX model
"""
import numpy as np
from PIL import Image
import onnxruntime as ort
import sys

def preprocess_image(image_path):
    """Preprocess image (matches training)."""
    image = Image.open(image_path).convert('RGB')

    print(f"Original size: {image.size}")

    # Resize shortest side to 256 (maintaining aspect ratio)
    w, h = image.size
    if w < h:
        new_w, new_h = 256, int(256 * h / w)
    else:
        new_h, new_w = 256, int(256 * w / h)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    print(f"After resize: {image.size}")

    # Center crop to 224x224
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    print(f"After crop: {image.size}")

    # Normalize
    image_np = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_np = (image_np - mean) / std

    # CHW format + batch dimension
    image_np = np.transpose(image_np, (2, 0, 1))
    image_np = np.expand_dims(image_np, axis=0)

    return image_np

# Load model
model_path = "models/vl_distilled_resnet18/model_quantized.onnx"
print(f"Loading model: {model_path}")
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# Test image
image_path = sys.argv[1] if len(sys.argv) > 1 else "graph.png"
print(f"\nTesting image: {image_path}")

# Preprocess
img_tensor = preprocess_image(image_path)
print(f"Tensor shape: {img_tensor.shape}")

# Inference
outputs = session.run(['output'], {'input': img_tensor})
logits = outputs[0][0]

# Softmax
probs = np.exp(logits) / np.sum(np.exp(logits))
pred = np.argmax(probs)
class_name = "LOW" if pred == 0 else "HIGH"

print(f"\n{'='*60}")
print(f"Prediction: {class_name} ({probs[pred]*100:.1f}%)")
print(f"Probabilities: LOW={probs[0]:.3f}, HIGH={probs[1]:.3f}")
print(f"Logits: LOW={logits[0]:.3f}, HIGH={logits[1]:.3f}")
print(f"{'='*60}")
