"""
Quick Model Comparison Test (No MTEB required)
Tests model dimensions, speed, and similarity

Usage:
    python scripts/quick_test.py
    python scripts/quick_test.py --model qwen3-m2v
"""

import sys
from pathlib import Path
import time
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model2vec import StaticModel
import numpy as np
from typing import List

# Test sentences
TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "A fast brown fox leaps above a sleepy canine",
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Paris is the capital city of France",
    "London is the capital of the United Kingdom",
    "Python is a popular programming language",
    "JavaScript is widely used for web development",
    "The weather is sunny and warm today",
    "It's a beautiful day with clear blue skies",
]

MODELS = {
    "gemma-768d": "tss-deposium/gemma-deposium-768d",
    "qwen3-m2v": "Pringled/m2v-Qwen3-Embedding-0.6B",
    "int8": "C10X/int8",
}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_model(model_key: str, verbose: bool = True):
    """
    Test a model's performance and quality

    Returns:
        dict with metrics
    """
    if model_key not in MODELS:
        print(f"❌ Unknown model: {model_key}")
        return None

    model_name = MODELS[model_key]

    if verbose:
        print(f"\n{'='*70}")
        print(f"🧪 Testing: {model_key}")
        print(f"   Source: {model_name}")
        print(f"{'='*70}")

    # Load model
    print(f"\n⏳ Loading model...")
    load_start = time.time()
    model = StaticModel.from_pretrained(model_name)
    load_time = time.time() - load_start
    print(f"✅ Loaded in {load_time:.2f}s")

    # Test dimensions
    test_embedding = model.encode(["test"], show_progress_bar=False)[0]
    dims = len(test_embedding)
    print(f"📐 Dimensions: {dims}D")

    # Test speed
    print(f"\n⚡ Speed test ({len(TEST_SENTENCES)} sentences)...")

    # Warmup
    _ = model.encode(TEST_SENTENCES[:2], show_progress_bar=False)

    # Actual timing
    start = time.time()
    embeddings = model.encode(TEST_SENTENCES, show_progress_bar=False)
    elapsed = time.time() - start

    sentences_per_sec = len(TEST_SENTENCES) / elapsed
    ms_per_sentence = (elapsed / len(TEST_SENTENCES)) * 1000

    print(f"✅ Speed: {sentences_per_sec:.1f} sentences/sec ({ms_per_sentence:.1f} ms/sentence)")

    # Test semantic similarity
    print(f"\n🔍 Semantic Similarity Tests:")

    # Similar sentences (should be high similarity)
    sim1 = cosine_similarity(embeddings[0], embeddings[1])  # fox sentences
    sim2 = cosine_similarity(embeddings[2], embeddings[3])  # ML sentences
    sim3 = cosine_similarity(embeddings[4], embeddings[5])  # capital cities
    sim4 = cosine_similarity(embeddings[6], embeddings[7])  # programming
    sim5 = cosine_similarity(embeddings[8], embeddings[9])  # weather

    # Dissimilar sentences (should be low similarity)
    dissim1 = cosine_similarity(embeddings[0], embeddings[4])  # fox vs Paris
    dissim2 = cosine_similarity(embeddings[2], embeddings[6])  # ML vs Python

    similar_avg = np.mean([sim1, sim2, sim3, sim4, sim5])
    dissimilar_avg = np.mean([dissim1, dissim2])

    print(f"  Similar pairs:")
    print(f"    Fox sentences:     {sim1:.4f}")
    print(f"    ML sentences:      {sim2:.4f}")
    print(f"    Capital cities:    {sim3:.4f}")
    print(f"    Programming:       {sim4:.4f}")
    print(f"    Weather:           {sim5:.4f}")
    print(f"    Average (similar): {similar_avg:.4f} ✅")

    print(f"\n  Dissimilar pairs:")
    print(f"    Fox vs Paris:      {dissim1:.4f}")
    print(f"    ML vs Python:      {dissim2:.4f}")
    print(f"    Average (dissim):  {dissimilar_avg:.4f}")

    # Separation score (higher is better)
    separation = similar_avg - dissimilar_avg
    print(f"\n  📊 Separation Score: {separation:.4f} (higher = better discrimination)")

    # Estimate model size
    try:
        from huggingface_hub import model_info
        info = model_info(model_name)
        size_mb = info.siblings[0].size / (1024 * 1024) if info.siblings else "Unknown"
        print(f"\n💾 Model Size: {size_mb:.1f} MB (estimated)" if isinstance(size_mb, float) else "\n💾 Model Size: Unknown")
    except:
        print(f"\n💾 Model Size: Not available")

    results = {
        "model": model_key,
        "dimensions": dims,
        "load_time_s": load_time,
        "speed_sentences_per_sec": sentences_per_sec,
        "ms_per_sentence": ms_per_sentence,
        "similarity_similar": similar_avg,
        "similarity_dissimilar": dissimilar_avg,
        "separation_score": separation,
    }

    return results


def compare_all_models():
    """Compare all available models"""
    print(f"\n{'='*70}")
    print(f"🏆 COMPARING ALL MODELS")
    print(f"{'='*70}")

    all_results = []
    for model_key in MODELS.keys():
        try:
            results = test_model(model_key, verbose=True)
            if results:
                all_results.append(results)
        except Exception as e:
            print(f"❌ Failed to test {model_key}: {e}")

    # Summary comparison
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"📊 SUMMARY COMPARISON")
        print(f"{'='*70}\n")

        print(f"{'Model':<15} {'Dims':>6} {'Speed (sent/s)':>15} {'Separation':>12}")
        print(f"{'-'*70}")

        for r in all_results:
            print(f"{r['model']:<15} {r['dimensions']:>6} {r['speed_sentences_per_sec']:>15.1f} {r['separation_score']:>12.4f}")

        # Winner
        best_speed = max(all_results, key=lambda x: x['speed_sentences_per_sec'])
        best_separation = max(all_results, key=lambda x: x['separation_score'])

        print(f"\n🏆 Fastest: {best_speed['model']} ({best_speed['speed_sentences_per_sec']:.1f} sent/s)")
        print(f"🎯 Best Separation: {best_separation['model']} ({best_separation['separation_score']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Quick model comparison test")
    parser.add_argument("--model", type=str, choices=list(MODELS.keys()),
                      help="Test a specific model")
    parser.add_argument("--all", action="store_true",
                      help="Compare all models")

    args = parser.parse_args()

    if args.all or not args.model:
        compare_all_models()
    else:
        test_model(args.model, verbose=True)


if __name__ == "__main__":
    main()
