"""
Compare old vs new VL model on graph.png
"""
import numpy as np
from PIL import Image
import onnxruntime as ort

def preprocess_image(image_path):
    """Preprocess image (matches training)."""
    image = Image.open(image_path).convert('RGB')

    # Resize shortest side to 256 (maintaining aspect ratio)
    w, h = image.size
    if w < h:
        new_w, new_h = 256, int(256 * h / w)
    else:
        new_h, new_w = 256, int(256 * w / h)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    # Center crop to 224x224
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    # Normalize
    image_np = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_np = (image_np - mean) / std

    # CHW format + batch dimension
    image_np = np.transpose(image_np, (2, 0, 1))
    image_np = np.expand_dims(image_np, axis=0)

    return image_np

def test_model(model_path, model_name):
    """Test a model on graph.png"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    # Load model
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # Preprocess
    img_tensor = preprocess_image("graph.png")

    # Inference
    outputs = session.run(['output'], {'input': img_tensor})
    logits = outputs[0][0]

    # Softmax
    probs = np.exp(logits) / np.sum(np.exp(logits))
    pred = np.argmax(probs)
    class_name = "LOW" if pred == 0 else "HIGH"

    print(f"Prediction: {class_name} ({probs[pred]*100:.1f}%)")
    print(f"Probabilities: LOW={probs[0]:.3f}, HIGH={probs[1]:.3f}")
    print(f"Logits: LOW={logits[0]:.3f}, HIGH={logits[1]:.3f}")

    return class_name, probs[pred], probs

# Test old model
old_class, old_conf, old_probs = test_model("/tmp/old_vl_model.onnx", "OLD MODEL (fe91f03)")

# Test new model
new_class, new_conf, new_probs = test_model("models/vl_distilled_resnet18/model_quantized.onnx", "NEW MODEL (ResNet18 distilled from CLIP)")

# Comparison
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"Old model: {old_class} ({old_conf*100:.1f}%)")
print(f"New model: {new_class} ({new_conf*100:.1f}%)")
print(f"\nDifference in HIGH probability: {(new_probs[1] - old_probs[1])*100:+.1f}%")

if old_class == new_class:
    print(f"\n✅ Both models agree: {old_class}")
    if new_conf > old_conf:
        print(f"🎯 New model is MORE confident (+{(new_conf - old_conf)*100:.1f}%)")
    else:
        print(f"⚠️  Old model was MORE confident (+{(old_conf - new_conf)*100:.1f}%)")
else:
    print(f"\n⚠️  DISAGREEMENT! Old={old_class}, New={new_class}")
