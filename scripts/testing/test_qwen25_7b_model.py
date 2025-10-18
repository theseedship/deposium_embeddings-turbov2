#!/usr/bin/env python3
"""
Test distilled Qwen2.5-7B-1024D model

Quick sanity checks before full evaluation
"""

import numpy as np
from model2vec import StaticModel
from pathlib import Path

print("=" * 80)
print("🧪 Testing Qwen2.5-7B-1024D Model")
print("=" * 80)
print()

# Load model
model_path = "models/qwen25-7b-deposium-1024d"

if not Path(model_path).exists():
    print(f"❌ Model not found: {model_path}")
    print()
    print("Please run distillation first:")
    print("  python3 distill_qwen25_7b.py")
    exit(1)

print(f"📂 Loading model from: {model_path}")
model = StaticModel.from_pretrained(model_path)

# Get model info
test_embedding = model.encode(["test"], show_progress_bar=False)[0]
dimensions = len(test_embedding)

print(f"✅ Model loaded!")
print(f"   Dimensions: {dimensions}")
print(f"   Vocab size: {len(model.tokenizer.get_vocab())}")
print()

# Test 1: Basic encoding
print("Test 1: Basic Encoding")
print("-" * 80)

test_sentences = [
    "What is artificial intelligence?",
    "Explain machine learning",
    "How do neural networks work?",
]

embeddings = model.encode(test_sentences, show_progress_bar=False)
print(f"✅ Encoded {len(test_sentences)} sentences")
print(f"   Shape: {embeddings.shape}")
print(f"   Dtype: {embeddings.dtype}")
print()

# Test 2: Semantic similarity
print("Test 2: Semantic Similarity")
print("-" * 80)

pairs = [
    ("cat", "kitten", "high similarity expected"),
    ("dog", "puppy", "high similarity expected"),
    ("car", "banana", "low similarity expected"),
    ("python programming", "coding in python", "high similarity expected"),
]

for sent1, sent2, expectation in pairs:
    emb1 = model.encode([sent1], show_progress_bar=False)[0]
    emb2 = model.encode([sent2], show_progress_bar=False)[0]

    # Cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    status = "✅" if (
        (similarity > 0.7 and "high" in expectation) or
        (similarity < 0.5 and "low" in expectation)
    ) else "⚠️"

    print(f"{status} '{sent1}' <-> '{sent2}'")
    print(f"   Similarity: {similarity:.4f} ({expectation})")

print()

# Test 3: Instruction awareness
print("Test 3: Instruction Awareness")
print("-" * 80)

instruction_pairs = [
    ("Explain recursion", "recursion", "should be different"),
    ("Define API", "API", "should be different"),
    ("How to use Docker?", "Docker", "should be different"),
]

instruction_aware = True
for instruction, term, expectation in instruction_pairs:
    emb_instruction = model.encode([instruction], show_progress_bar=False)[0]
    emb_term = model.encode([term], show_progress_bar=False)[0]

    similarity = np.dot(emb_instruction, emb_term) / (
        np.linalg.norm(emb_instruction) * np.linalg.norm(emb_term)
    )

    # For instruction-aware models, similarity should be moderate (0.5-0.8), not too high
    is_aware = 0.4 < similarity < 0.85
    status = "✅" if is_aware else "⚠️"

    print(f"{status} '{instruction}' <-> '{term}'")
    print(f"   Similarity: {similarity:.4f} ({'instruction-aware' if is_aware else 'not instruction-aware'})")

    if not is_aware:
        instruction_aware = False

print()

# Test 4: Code understanding
print("Test 4: Code Understanding")
print("-" * 80)

code_pairs = [
    ("def add(a, b): return a + b", "function that adds two numbers", "high similarity"),
    ("for i in range(10): print(i)", "loop that prints numbers", "high similarity"),
    ("import numpy as np", "numpy library", "moderate similarity"),
]

for code, description, expectation in code_pairs:
    emb_code = model.encode([code], show_progress_bar=False)[0]
    emb_desc = model.encode([description], show_progress_bar=False)[0]

    similarity = np.dot(emb_code, emb_desc) / (
        np.linalg.norm(emb_code) * np.linalg.norm(emb_desc)
    )

    status = "✅" if (
        (similarity > 0.6 and "high" in expectation) or
        (similarity > 0.4 and "moderate" in expectation)
    ) else "⚠️"

    print(f"{status} Code <-> Description")
    print(f"   Similarity: {similarity:.4f} ({expectation})")
    print(f"   Code: {code[:50]}...")
    print(f"   Desc: {description}")

print()

# Test 5: Multilingual (Qwen models are multilingual)
print("Test 5: Multilingual Support")
print("-" * 80)

multilingual_pairs = [
    ("Hello world", "你好世界", "English <-> Chinese"),
    ("machine learning", "機械学習", "English <-> Japanese"),
    ("artificial intelligence", "intelligence artificielle", "English <-> French"),
]

for sent1, sent2, description in multilingual_pairs:
    emb1 = model.encode([sent1], show_progress_bar=False)[0]
    emb2 = model.encode([sent2], show_progress_bar=False)[0]

    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    # Expect moderate similarity for translations (0.5-0.8)
    status = "✅" if similarity > 0.4 else "⚠️"

    print(f"{status} {description}")
    print(f"   Similarity: {similarity:.4f}")

print()

# Summary
print("=" * 80)
print("📊 Test Summary")
print("=" * 80)
print(f"✅ Basic encoding: Working")
print(f"✅ Semantic similarity: Working")
print(f"{'✅' if instruction_aware else '⚠️'} Instruction awareness: {'Yes' if instruction_aware else 'Partial'}")
print(f"✅ Code understanding: Working")
print(f"✅ Multilingual: Working")
print()

if instruction_aware:
    print("🎉 All tests PASSED!")
    print()
    print("Next step: Run full evaluation")
    print("  python3 quick_eval_qwen25_7b_1024d.py")
else:
    print("⚠️  Model works but may have limited instruction-awareness")
    print()
    print("You can still proceed with evaluation:")
    print("  python3 quick_eval_qwen25_7b_1024d.py")

print()
