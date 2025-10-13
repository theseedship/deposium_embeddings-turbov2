"""
Test Reranking Functionality
Tests Qwen3-Embedding-0.6B vs int8 for reranking quality

Usage:
    python scripts/test_reranking.py
    python scripts/test_reranking.py --model qwen3-rerank
    python scripts/test_reranking.py --compare
"""

import requests
import argparse
import time
from typing import List, Dict

# FastAPI service URL (adjust if needed)
API_URL = "http://localhost:11435"

# Test data: Query + Documents (relevant and irrelevant mixed)
TEST_CASES = [
    {
        "query": "What is the capital of France?",
        "documents": [
            "Paris is the capital and largest city of France.",  # RELEVANT
            "Berlin is the capital of Germany.",
            "The Eiffel Tower is located in Paris, France.",  # RELEVANT
            "London is the capital of the United Kingdom.",
            "France is a country in Western Europe with Paris as its capital.",  # RELEVANT
            "Pizza is a popular Italian dish.",
            "Python is a programming language.",
        ],
        "expected_top_indices": [0, 4, 2],  # Expected most relevant documents
    },
    {
        "query": "How to train a machine learning model?",
        "documents": [
            "Machine learning models require data, algorithms, and training.",  # RELEVANT
            "The weather is sunny today.",
            "To train a model, split your data into train and test sets.",  # RELEVANT
            "Coffee is made from coffee beans.",
            "Deep learning is a subset of machine learning using neural networks.",  # RELEVANT
            "The cat sat on the mat.",
            "Model training involves iterative optimization of parameters.",  # RELEVANT
        ],
        "expected_top_indices": [0, 6, 2, 4],
    },
    {
        "query": "Best restaurants in Tokyo",
        "documents": [
            "Tokyo has many excellent sushi restaurants.",  # RELEVANT
            "The Great Wall of China is a historic landmark.",
            "Ramen shops are very popular in Tokyo.",  # RELEVANT
            "Machine learning requires computational resources.",
            "Tokyo is known for its diverse and high-quality food scene.",  # RELEVANT
            "Python uses indentation for code blocks.",
        ],
        "expected_top_indices": [0, 4, 2],
    },
]


def test_rerank(model: str, query: str, documents: List[str]) -> Dict:
    """
    Test reranking with the specified model

    Returns:
        dict with results, timing, and scores
    """
    url = f"{API_URL}/api/rerank"

    payload = {
        "model": model,
        "query": query,
        "documents": documents,
    }

    start = time.time()
    response = requests.post(url, json=payload)
    elapsed = time.time() - start

    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

    result = response.json()

    return {
        "model": model,
        "results": result["results"],
        "time_ms": elapsed * 1000,
        "top_3_indices": [r["index"] for r in result["results"][:3]],
        "top_3_scores": [r["relevance_score"] for r in result["results"][:3]],
    }


def evaluate_ranking(test_case: Dict, results: Dict) -> Dict:
    """
    Evaluate ranking quality using NDCG-like metric

    Returns:
        dict with metrics
    """
    predicted = results["top_3_indices"]
    expected = test_case["expected_top_indices"][:3]

    # Calculate precision@3 (how many of top-3 are actually relevant)
    hits = len(set(predicted) & set(expected))
    precision_at_3 = hits / 3.0

    # Check if #1 result is in expected top results
    top_1_correct = predicted[0] in expected

    return {
        "precision_at_3": precision_at_3,
        "top_1_correct": top_1_correct,
        "hits": hits,
    }


def run_single_test(model: str = "qwen3-rerank"):
    """Test a single model"""
    print(f"\n{'='*70}")
    print(f"🧪 Testing Reranking: {model}")
    print(f"{'='*70}\n")

    total_precision = 0
    total_top1_correct = 0

    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n📋 Test Case {i}: {test_case['query']}")
        print(f"   Documents: {len(test_case['documents'])}")

        try:
            result = test_rerank(model, test_case["query"], test_case["documents"])
            metrics = evaluate_ranking(test_case, result)

            print(f"\n✅ Results ({result['time_ms']:.1f} ms):")
            print(f"   Top 3 indices: {result['top_3_indices']}")
            print(f"   Top 3 scores:  {[f'{s:.4f}' for s in result['top_3_scores']]}")
            print(f"\n📊 Metrics:")
            print(f"   Precision@3:   {metrics['precision_at_3']:.2f} ({metrics['hits']}/3)")
            print(f"   Top-1 Correct: {'✅' if metrics['top_1_correct'] else '❌'}")

            # Show top-ranked documents
            print(f"\n🔝 Top 3 Documents:")
            for rank, res in enumerate(result["results"][:3], 1):
                doc = res["document"][:70] + "..." if len(res["document"]) > 70 else res["document"]
                print(f"   {rank}. [{res['relevance_score']:.4f}] {doc}")

            total_precision += metrics["precision_at_3"]
            total_top1_correct += 1 if metrics["top_1_correct"] else 0

        except Exception as e:
            print(f"❌ Error: {e}")

    # Summary
    avg_precision = total_precision / len(TEST_CASES)
    top1_accuracy = total_top1_correct / len(TEST_CASES)

    print(f"\n{'='*70}")
    print(f"📊 SUMMARY for {model}")
    print(f"{'='*70}")
    print(f"Average Precision@3: {avg_precision:.2f}")
    print(f"Top-1 Accuracy:      {top1_accuracy:.2f}")
    print(f"Test Cases:          {len(TEST_CASES)}")


def compare_models():
    """Compare qwen3-rerank vs int8"""
    print(f"\n{'='*70}")
    print(f"🏆 COMPARING RERANKING MODELS")
    print(f"{'='*70}\n")

    models = ["qwen3-rerank", "int8"]
    results = {}

    for model in models:
        print(f"\n{'='*70}")
        print(f"Testing: {model}")
        print(f"{'='*70}")

        model_results = {
            "precision": 0,
            "top1_correct": 0,
            "total_time_ms": 0,
        }

        for i, test_case in enumerate(TEST_CASES, 1):
            print(f"\n📋 Test {i}/{len(TEST_CASES)}: {test_case['query'][:50]}...")

            try:
                result = test_rerank(model, test_case["query"], test_case["documents"])
                metrics = evaluate_ranking(test_case, result)

                model_results["precision"] += metrics["precision_at_3"]
                model_results["top1_correct"] += 1 if metrics["top_1_correct"] else 0
                model_results["total_time_ms"] += result["time_ms"]

                print(f"   P@3: {metrics['precision_at_3']:.2f} | Top-1: {'✅' if metrics['top_1_correct'] else '❌'} | Time: {result['time_ms']:.1f}ms")

            except Exception as e:
                print(f"   ❌ Error: {e}")

        # Calculate averages
        model_results["avg_precision"] = model_results["precision"] / len(TEST_CASES)
        model_results["top1_accuracy"] = model_results["top1_correct"] / len(TEST_CASES)
        model_results["avg_time_ms"] = model_results["total_time_ms"] / len(TEST_CASES)

        results[model] = model_results

    # Comparison table
    print(f"\n{'='*70}")
    print(f"📊 COMPARISON RESULTS")
    print(f"{'='*70}\n")

    print(f"{'Model':<20} {'Precision@3':>12} {'Top-1 Acc':>12} {'Avg Time (ms)':>15}")
    print(f"{'-'*70}")

    for model, res in results.items():
        print(f"{model:<20} {res['avg_precision']:>12.2f} {res['top1_accuracy']:>12.2f} {res['avg_time_ms']:>15.1f}")

    # Winner
    best_precision = max(results.items(), key=lambda x: x[1]['avg_precision'])
    fastest = min(results.items(), key=lambda x: x[1]['avg_time_ms'])

    print(f"\n🏆 Best Precision@3: {best_precision[0]} ({best_precision[1]['avg_precision']:.2f})")
    print(f"⚡ Fastest:          {fastest[0]} ({fastest[1]['avg_time_ms']:.1f} ms)")


def main():
    parser = argparse.ArgumentParser(description="Test reranking functionality")
    parser.add_argument("--model", type=str, choices=["qwen3-rerank", "int8"],
                      default="qwen3-rerank", help="Model to test")
    parser.add_argument("--compare", action="store_true",
                      help="Compare both reranking models")

    args = parser.parse_args()

    # Check if API is available
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ API not healthy: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API at {API_URL}: {e}")
        print(f"   Make sure the embeddings service is running on port 11435")
        return

    if args.compare:
        compare_models()
    else:
        run_single_test(args.model)


if __name__ == "__main__":
    main()
