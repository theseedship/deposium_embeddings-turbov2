#!/usr/bin/env python3
"""
Check ONNX Model Output Structure
"""

from pathlib import Path
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_path = Path("./models/gemma-onnx-int8")

print("Loading model and tokenizer...")
model = ORTModelForFeatureExtraction.from_pretrained(
    str(model_path),
    provider="CPUExecutionProvider"
)
tokenizer = AutoTokenizer.from_pretrained(str(model_path))

print("\nTesting with sample text...")
text = "This is a test"
inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)

print(f"\nOutput type: {type(outputs)}")
print(f"\nOutput keys: {outputs.keys() if hasattr(outputs, 'keys') else 'No keys (not a dict)'}")
print(f"\nOutput structure:")
for key, value in outputs.items():
    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
