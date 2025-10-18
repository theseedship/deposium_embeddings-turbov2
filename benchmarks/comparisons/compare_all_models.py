#!/usr/bin/env python3
"""
Compare ALL embedding models with real-world functionality tests
Tests: Quality, Instruction-awareness, Speed, Size
"""

import time
import numpy as np
from pathlib import Path
from model2vec import StaticModel

print("=" * 80)
print("🔬 COMPLETE MODEL COMPARISON - Real-world Functionality Tests")
print("=" * 80)
print()

# Load all models
models = {}
model_paths = {
    "Qwen2.5-1.5B (CURRENT)": "models/qwen25-deposium-1024d",
    "Qwen2.5-3B (NEW)": "qwen25-3b-deposium-1024d",
}

print("📥 Loading models...")
for name, path in model_paths.items():
    model_dir = Path(path)
    if model_dir.exists():
        try:
            print(f"  Loading {name}...")
            models[name] = StaticModel.from_pretrained(str(model_dir))

            # Get size
            size_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
            size_mb = size_bytes / (1024 * 1024)
            print(f"    ✅ Loaded ({size_mb:.1f}MB)")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
    else:
        print(f"  ❌ Not found: {path}")

print()
print("=" * 80)
print("TEST 1: SEMANTIC SIMILARITY (Quality Test)")
print("=" * 80)
print()

# Test semantic similarity
test_pairs = [
    ("Machine learning is fascinating", "AI and deep learning are interesting"),
    ("The weather is nice today", "It's sunny and warm outside"),
    ("Python programming language", "Coding in Python"),
    ("Database optimization techniques", "SQL query performance tuning"),
    ("Climate change impacts", "Global warming effects"),
]

print("Test: Measure semantic similarity between related sentences")
print()

similarity_scores = {name: [] for name in models}

for text1, text2 in test_pairs:
    print(f"Pair: '{text1[:40]}...' vs '{text2[:40]}...'")

    for name, model in models.items():
        emb1 = model.encode([text1])[0]
        emb2 = model.encode([text2])[0]

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarity_scores[name].append(similarity)

        print(f"  {name}: {similarity:.4f}")
    print()

# Average similarity
print("📊 Average Semantic Similarity:")
for name in models:
    avg_sim = np.mean(similarity_scores[name])
    print(f"  {name}: {avg_sim:.4f} ({avg_sim*100:.2f}%)")

print()
print("=" * 80)
print("TEST 2: INSTRUCTION-AWARENESS (Unique Model2Vec capability)")
print("=" * 80)
print()

# Test instruction understanding
instruction_tests = [
    ("Explain quantum computing", "quantum computing"),
    ("Find restaurants near me", "restaurant"),
    ("Compare iPhone vs Android", "iPhone Android comparison"),
    ("Summarize climate change", "climate change summary"),
    ("Translate hello to French", "hello French translation"),
]

print("Test: Can model distinguish instruction from topic?")
print("Higher score = better instruction-awareness")
print()

instruction_scores = {name: [] for name in models}

for instruction, topic in instruction_tests:
    print(f"Instruction: '{instruction}' vs Topic: '{topic}'")

    for name, model in models.items():
        # Encode instruction and pure topic
        emb_instruction = model.encode([instruction])[0]
        emb_topic = model.encode([topic])[0]

        # Measure difference (lower similarity = better instruction-awareness)
        similarity = np.dot(emb_instruction, emb_topic) / (np.linalg.norm(emb_instruction) * np.linalg.norm(emb_topic))

        # Instruction awareness score: 1 - similarity (higher = better)
        awareness_score = 1 - similarity
        instruction_scores[name].append(awareness_score)

        print(f"  {name}: {awareness_score:.4f} (sim: {similarity:.4f})")
    print()

print("📊 Average Instruction-Awareness Score:")
for name in models:
    avg_score = np.mean(instruction_scores[name])
    print(f"  {name}: {avg_score:.4f} ({avg_score*100:.2f}%)")

print()
print("=" * 80)
print("TEST 3: SPEED BENCHMARK (Inference performance)")
print("=" * 80)
print()

# Speed test
speed_texts = [
    "Machine learning is transforming industries",
    "Natural language processing enables AI",
    "Deep learning models require GPUs",
    "Python is popular for data science",
] * 25  # 100 texts total

print(f"Test: Encode {len(speed_texts)} texts and measure time")
print()

speed_results = {}

for name, model in models.items():
    print(f"Testing {name}...")

    start_time = time.time()
    embeddings = model.encode(speed_texts)
    end_time = time.time()

    duration = end_time - start_time
    texts_per_sec = len(speed_texts) / duration

    speed_results[name] = {
        "duration": duration,
        "texts_per_sec": texts_per_sec,
    }

    print(f"  Duration: {duration:.3f}s")
    print(f"  Throughput: {texts_per_sec:.1f} texts/sec")
    print()

print()
print("=" * 80)
print("TEST 4: DOCUMENT RETRIEVAL (RAG simulation)")
print("=" * 80)
print()

# Document retrieval test
documents = [
    "Python is a high-level programming language",
    "Machine learning algorithms learn from data",
    "Databases store and organize information",
    "Web servers handle HTTP requests",
    "Neural networks mimic brain structure",
    "Cloud computing provides scalable resources",
    "Cybersecurity protects digital assets",
]

queries = [
    "What is Python used for?",
    "How does machine learning work?",
    "Tell me about databases",
]

print("Test: Retrieve relevant documents for queries (RAG use case)")
print()

retrieval_scores = {name: [] for name in models}

for query in queries:
    print(f"Query: '{query}'")

    for name, model in models.items():
        # Encode query and documents
        query_emb = model.encode([query])[0]
        doc_embs = model.encode(documents)

        # Calculate similarities
        similarities = []
        for doc_emb in doc_embs:
            sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
            similarities.append(sim)

        # Get top result
        top_idx = np.argmax(similarities)
        top_sim = similarities[top_idx]

        retrieval_scores[name].append(top_sim)

        print(f"  {name}: Top match = '{documents[top_idx][:50]}...' (sim: {top_sim:.4f})")
    print()

print("📊 Average Retrieval Quality:")
for name in models:
    avg_score = np.mean(retrieval_scores[name])
    print(f"  {name}: {avg_score:.4f} ({avg_score*100:.2f}%)")

print()
print("=" * 80)
print("📊 FINAL COMPARISON SUMMARY")
print("=" * 80)
print()

# Create comparison table
print(f"{'Model':<30} {'Quality':<12} {'Instruct':<12} {'Speed':<15} {'Size':<10}")
print("-" * 80)

for name in models:
    quality = np.mean(similarity_scores[name]) * 100
    instruct = np.mean(instruction_scores[name]) * 100
    speed = speed_results[name]["texts_per_sec"]

    # Get size
    path = model_paths[name]
    model_dir = Path(path)
    size_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    size_mb = size_bytes / (1024 * 1024)

    print(f"{name:<30} {quality:>6.2f}%     {instruct:>6.2f}%     {speed:>8.1f} t/s    {size_mb:>6.1f}MB")

print()
print("=" * 80)
print("🏆 RECOMMENDATION")
print("=" * 80)
print()

# Determine winner
quality_3b = np.mean(similarity_scores.get("Qwen2.5-3B (NEW)", [0])) * 100
quality_1_5b = np.mean(similarity_scores.get("Qwen2.5-1.5B (CURRENT)", [0])) * 100

if quality_3b > quality_1_5b:
    improvement = quality_3b - quality_1_5b
    print(f"✅ UPGRADE TO QWEN2.5-3B")
    print(f"   Quality improvement: +{improvement:.2f}%")
    print(f"   From {quality_1_5b:.2f}% → {quality_3b:.2f}%")
    print()
    print(f"   Qwen2.5-3B achieves {quality_3b:.2f}% quality")
    if quality_3b >= 85:
        print(f"   ✅ Exceeds 85% target - READY FOR PRODUCTION")
    else:
        print(f"   ⚠️  Below 85% target - Consider waiting for 7B model")
else:
    print(f"⚠️  NO UPGRADE RECOMMENDED")
    print(f"   Qwen2.5-3B: {quality_3b:.2f}%")
    print(f"   Current 1.5B: {quality_1_5b:.2f}%")
    print(f"   Wait for Qwen2.5-7B for better results")

print()
print("=" * 80)
print("Next Steps:")
print("=" * 80)
print()
if quality_3b >= 85:
    print("1. ✅ Deploy Qwen2.5-3B to production")
    print("2. Update API to use new model")
    print("3. Update Docker image")
    print("4. Monitor performance in production")
else:
    print("1. ⏳ Wait for Qwen2.5-7B distillation on HuggingFace")
    print("2. Expected: 91-95% quality (vs current 3B: {quality_3b:.2f}%)")
    print("3. Keep current 1.5B model for now")

print()
