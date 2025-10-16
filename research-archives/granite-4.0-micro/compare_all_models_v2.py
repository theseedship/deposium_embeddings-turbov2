#!/usr/bin/env python3
"""
Compare ALL embedding models with real-world functionality tests
Models: Qwen2.5-1.5B, Qwen2.5-3B, Granite 4.0 Micro, Gemma-768D
Tests: Quality, Instruction-awareness, Speed, Size, Multilingual
"""

import time
import numpy as np
from pathlib import Path
from model2vec import StaticModel

print("=" * 80)
print("🔬 COMPLETE MODEL COMPARISON - 4 Models + Multilingual Tests")
print("=" * 80)
print()

# Load all models
models = {}
model_paths = {
    "Qwen2.5-1.5B (PROD)": "models/qwen25-deposium-1024d",
    "Qwen2.5-3B": "qwen25-3b-deposium-1024d",
    "Granite 4.0 Micro (NEW)": "granite-4.0-micro-deposium-1024d",
    "Gemma-768D": "models/gemma-deposium-768d",
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
        print(f"  ⚠️  Not found: {path}")

if not models:
    print()
    print("❌ No models loaded! Exiting.")
    exit(1)

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
print("TEST 5: MULTILINGUAL CAPABILITY (🌍 NEW TEST)")
print("=" * 80)
print()

# Multilingual test
multilingual_tests = [
    ("en", "Machine learning is transforming AI", "AI and machine learning are revolutionary"),
    ("fr", "Le deep learning révolutionne l'IA", "L'IA et le deep learning sont révolutionnaires"),
    ("de", "Das maschinelle Lernen verändert KI", "KI und maschinelles Lernen sind revolutionär"),
    ("es", "El aprendizaje automático transforma la IA", "La IA y el aprendizaje automático son revolucionarios"),
    ("zh", "机器学习正在改变人工智能", "人工智能和机器学习具有革命性"),
]

print("Test: Semantic similarity in multiple languages")
print("(Granite 4.0 Micro is natively multilingual - 12 languages)")
print()

multilingual_scores = {name: [] for name in models}

for lang, text1, text2 in multilingual_tests:
    print(f"[{lang.upper()}] '{text1[:40]}...' vs '{text2[:40]}...'")

    for name, model in models.items():
        try:
            emb1 = model.encode([text1])[0]
            emb2 = model.encode([text2])[0]

            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            multilingual_scores[name].append(similarity)

            print(f"  {name}: {similarity:.4f}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
            multilingual_scores[name].append(0.0)
    print()

print("📊 Average Multilingual Quality:")
for name in models:
    if multilingual_scores[name]:
        avg_score = np.mean(multilingual_scores[name])
        print(f"  {name}: {avg_score:.4f} ({avg_score*100:.2f}%)")

print()
print("=" * 80)
print("📊 FINAL COMPARISON SUMMARY")
print("=" * 80)
print()

# Create comparison table
print(f"{'Model':<25} {'Quality':<10} {'Instruct':<10} {'Multilng':<10} {'Speed':<12} {'Size':<10}")
print("-" * 90)

for name in models:
    quality = np.mean(similarity_scores[name]) * 100 if similarity_scores[name] else 0
    instruct = np.mean(instruction_scores[name]) * 100 if instruction_scores[name] else 0
    multilng = np.mean(multilingual_scores[name]) * 100 if multilingual_scores[name] else 0
    speed = speed_results[name]["texts_per_sec"] if name in speed_results else 0

    # Get size
    path = model_paths[name]
    model_dir = Path(path)
    if model_dir.exists():
        size_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        size_mb = size_bytes / (1024 * 1024)
    else:
        size_mb = 0

    print(f"{name:<25} {quality:>6.2f}%   {instruct:>6.2f}%   {multilng:>6.2f}%   {speed:>8.1f} t/s  {size_mb:>6.1f}MB")

print()
print("=" * 80)
print("🏆 RECOMMENDATION")
print("=" * 80)
print()

# Determine winner
current_quality = np.mean(similarity_scores.get("Qwen2.5-1.5B (PROD)", [0])) * 100
granite_quality = np.mean(similarity_scores.get("Granite 4.0 Micro (NEW)", [0])) * 100
granite_multilng = np.mean(multilingual_scores.get("Granite 4.0 Micro (NEW)", [0])) * 100

current_multilng = np.mean(multilingual_scores.get("Qwen2.5-1.5B (PROD)", [0])) * 100

print(f"📊 Key Metrics Comparison:")
print(f"   Qwen2.5-1.5B (CURRENT): {current_quality:.2f}% quality, {current_multilng:.2f}% multilingual")
print(f"   Granite 4.0 Micro (NEW): {granite_quality:.2f}% quality, {granite_multilng:.2f}% multilingual")
print()

if "Granite 4.0 Micro (NEW)" not in models:
    print("⚠️  GRANITE NOT LOADED - Complete distillation first")
    print("   Run: python3 distill_granite_4_0_micro.py")
elif granite_quality >= current_quality or granite_multilng > current_multilng + 5:
    improvement = granite_quality - current_quality
    multi_improvement = granite_multilng - current_multilng

    print(f"✅ UPGRADE TO GRANITE 4.0 MICRO RECOMMENDED")
    print()
    if improvement > 0:
        print(f"   Quality improvement: +{improvement:.2f}%")
        print(f"   From {current_quality:.2f}% → {granite_quality:.2f}%")
    if multi_improvement > 0:
        print(f"   Multilingual improvement: +{multi_improvement:.2f}%")
        print(f"   From {current_multilng:.2f}% → {granite_multilng:.2f}%")
    print()
    print(f"   🌍 Granite excels in multilingual support (12 languages)")
    print(f"   ✅ Modern architecture (GQA, RoPE, SwiGLU)")

    if granite_quality >= 88:
        print(f"   ✅ Quality {granite_quality:.2f}% is production-ready")
else:
    print(f"⚠️  KEEP CURRENT MODEL (Qwen2.5-1.5B)")
    print(f"   Current: {current_quality:.2f}% quality")
    print(f"   Granite: {granite_quality:.2f}% quality")
    print(f"   Difference: {granite_quality - current_quality:.2f}%")
    print()
    print(f"   Recommendation: Wait for Qwen2.5-7B (expected 91-95%)")

print()
print("=" * 80)
print("Next Steps:")
print("=" * 80)
print()

if granite_quality >= current_quality or granite_multilng > current_multilng + 5:
    print("1. ✅ Test Granite in production environment")
    print("2. Run python3 test_multilingual_granite.py for detailed multilingual analysis")
    print("3. Update API to use Granite model")
    print("4. Update Docker image")
    print("5. Monitor production performance")
else:
    print("1. ⏳ Archive Granite results for reference")
    print("2. Test other models: Llama 3.2 3B, Phi-3.5-mini, Mistral-Small")
    print("3. Or wait for Qwen2.5-7B on HuggingFace Spaces")

print()

# Save results to file
results_file = "granite_comparison_results.txt"
with open(results_file, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("GRANITE 4.0 MICRO COMPARISON RESULTS\n")
    f.write("=" * 80 + "\n\n")

    for name in models:
        quality = np.mean(similarity_scores[name]) * 100 if similarity_scores[name] else 0
        instruct = np.mean(instruction_scores[name]) * 100 if instruction_scores[name] else 0
        multilng = np.mean(multilingual_scores[name]) * 100 if multilingual_scores[name] else 0

        f.write(f"{name}:\n")
        f.write(f"  Quality: {quality:.2f}%\n")
        f.write(f"  Instruction-awareness: {instruct:.2f}%\n")
        f.write(f"  Multilingual: {multilng:.2f}%\n")
        f.write("\n")

print(f"📄 Results saved to: {results_file}")
print()
