"""
Benchmark: bge-reranker-v2-m3 (ONNX INT8) vs mxbai-rerank-v2 (PyTorch FP16)

Compares latency, quality (Precision@3, Top-1), and score spread.
Includes French + English + multilingual test cases.

Usage:
    python scripts/bench_reranker_compare.py
    python scripts/bench_reranker_compare.py --url https://deposiumembeddings-turbov2-production.up.railway.app
    python scripts/bench_reranker_compare.py --model bge-reranker-v2-m3
"""

import requests
import argparse
import time
import json
import statistics
from typing import List, Dict

DEFAULT_URL = "https://deposiumembeddings-turbov2-production.up.railway.app"

MODELS = ["bge-reranker-v2-m3", "mxbai-rerank-v2"]

# Benchmark test cases: mix of FR, EN, and multilingual
TEST_CASES = [
    # --- French (domain: real estate / notarial) ---
    {
        "id": "FR-1",
        "lang": "fr",
        "query": "Qu'est-ce que le droit de preemption urbain ?",
        "documents": [
            "Le droit de preemption urbain (DPU) permet a une collectivite locale d'acquerir en priorite un bien immobilier mis en vente dans un perimetre defini.",
            "La taxe fonciere est calculee sur la base de la valeur locative cadastrale du bien.",
            "Le PLU (Plan Local d'Urbanisme) definit les regles de construction applicables dans chaque zone de la commune.",
            "En cas de vente, le notaire doit purger le droit de preemption en adressant une DIA a la mairie.",
            "L'assurance habitation est obligatoire pour les locataires.",
            "Le bail commercial a une duree minimale de 9 ans.",
        ],
        "expected_top": [0, 3],
    },
    {
        "id": "FR-2",
        "lang": "fr",
        "query": "Comment calculer les frais de notaire pour un achat immobilier ?",
        "documents": [
            "Les frais de notaire comprennent les droits de mutation, les emoluments du notaire et les debours.",
            "Pour un bien ancien, les droits de mutation representent environ 5,8% du prix de vente.",
            "Le pret a taux zero (PTZ) est un dispositif d'aide a l'accession a la propriete.",
            "Les emoluments du notaire sont fixes par decret et calcules selon un bareme degressif.",
            "La loi Pinel permet une reduction d'impot pour l'investissement locatif dans le neuf.",
            "Le diagnostic de performance energetique (DPE) est obligatoire pour toute vente.",
        ],
        "expected_top": [0, 1, 3],
    },
    {
        "id": "FR-3",
        "lang": "fr",
        "query": "Quelles sont les obligations du vendeur lors d'une vente immobiliere ?",
        "documents": [
            "Le vendeur doit fournir les diagnostics techniques obligatoires (DPE, amiante, plomb, etc.).",
            "La blockchain est une technologie de registre distribue.",
            "Le vendeur a une obligation de delivrance conforme et doit garantir les vices caches.",
            "Les taux d'interet des credits immobiliers sont historiquement bas en 2024.",
            "Le compromis de vente engage les deux parties et prevoit generalement une clause de retractation de 10 jours.",
            "Python est un langage de programmation oriente objet.",
        ],
        "expected_top": [0, 2, 4],
    },
    # --- English ---
    {
        "id": "EN-1",
        "lang": "en",
        "query": "What is the capital of France?",
        "documents": [
            "Paris is the capital and largest city of France.",
            "Berlin is the capital of Germany.",
            "The Eiffel Tower is located in Paris, France.",
            "London is the capital of the United Kingdom.",
            "France is a country in Western Europe with Paris as its capital.",
            "Pizza is a popular Italian dish.",
            "Python is a programming language.",
        ],
        "expected_top": [0, 4, 2],
    },
    {
        "id": "EN-2",
        "lang": "en",
        "query": "How to train a machine learning model?",
        "documents": [
            "Machine learning models require data, algorithms, and training.",
            "The weather is sunny today.",
            "To train a model, split your data into train and test sets.",
            "Coffee is made from coffee beans.",
            "Deep learning is a subset of machine learning using neural networks.",
            "The cat sat on the mat.",
            "Model training involves iterative optimization of parameters.",
        ],
        "expected_top": [0, 6, 2, 4],
    },
    # --- Multilingual (query FR, docs mixed FR/EN) ---
    {
        "id": "MULTI-1",
        "lang": "multi",
        "query": "Comment fonctionne la signature electronique ?",
        "documents": [
            "La signature electronique a la meme valeur juridique que la signature manuscrite selon le reglement eIDAS.",
            "Electronic signatures use cryptographic algorithms to verify document authenticity and signer identity.",
            "Le cafe est une boisson populaire dans le monde entier.",
            "Digital certificates are issued by Certificate Authorities (CA) to validate electronic identities.",
            "La loi du 13 mars 2000 reconnait la preuve electronique en droit francais.",
            "The Great Wall of China is visible from space.",
        ],
        "expected_top": [0, 1, 4, 3],
    },
]


def rerank(url: str, model: str, query: str, documents: List[str]) -> Dict:
    """Call the rerank API and return results + timing."""
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
    }

    start = time.perf_counter()
    response = requests.post(f"{url}/api/rerank", json=payload, timeout=60)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if response.status_code != 200:
        return {"error": f"{response.status_code}: {response.text}", "time_ms": elapsed_ms}

    data = response.json()
    return {
        "results": data["results"],
        "time_ms": elapsed_ms,
        "top_indices": [r["index"] for r in data["results"]],
        "top_scores": [r["relevance_score"] for r in data["results"]],
    }


def evaluate(test_case: Dict, result: Dict) -> Dict:
    """Compute Precision@3, Top-1, and score spread."""
    if "error" in result:
        return {"error": result["error"]}

    predicted_top3 = result["top_indices"][:3]
    expected = test_case["expected_top"][:3]

    hits = len(set(predicted_top3) & set(expected))
    precision_at_3 = hits / min(3, len(expected))
    top1_correct = predicted_top3[0] in expected

    scores = result["top_scores"]
    spread = scores[0] - scores[-1] if len(scores) > 1 else 0

    return {
        "precision_at_3": precision_at_3,
        "top1_correct": top1_correct,
        "hits": hits,
        "spread": spread,
        "top_score": scores[0],
        "bottom_score": scores[-1],
    }


def run_benchmark(url: str, models: List[str], warmup: bool = True):
    """Run the full benchmark."""
    print(f"\n{'=' * 80}")
    print(f"  RERANKER BENCHMARK: {' vs '.join(models)}")
    print(f"  URL: {url}")
    print(f"  Test cases: {len(TEST_CASES)} ({sum(1 for t in TEST_CASES if t['lang']=='fr')} FR, "
          f"{sum(1 for t in TEST_CASES if t['lang']=='en')} EN, "
          f"{sum(1 for t in TEST_CASES if t['lang']=='multi')} Multi)")
    print(f"{'=' * 80}\n")

    # Health check
    try:
        r = requests.get(f"{url}/health", timeout=10)
        print(f"Health: {r.json()}\n")
    except Exception as e:
        print(f"WARNING: Health check failed: {e}\n")

    all_results = {}

    for model in models:
        print(f"\n{'─' * 80}")
        print(f"  Model: {model}")
        print(f"{'─' * 80}")

        # Warmup: first call loads the model (cold start)
        if warmup:
            print(f"  Warmup (cold start)...", end=" ", flush=True)
            warmup_result = rerank(url, model, "test query", ["test document"])
            if "error" in warmup_result:
                print(f"ERROR: {warmup_result['error']}")
                all_results[model] = {"error": warmup_result["error"]}
                continue
            print(f"{warmup_result['time_ms']:.0f}ms (model loading)")

        model_metrics = {
            "latencies": [],
            "precisions": [],
            "top1s": [],
            "spreads": [],
            "details": [],
        }

        for tc in TEST_CASES:
            result = rerank(url, model, tc["query"], tc["documents"])
            metrics = evaluate(tc, result)

            if "error" in metrics:
                print(f"  [{tc['id']}] ERROR: {metrics['error']}")
                continue

            model_metrics["latencies"].append(result["time_ms"])
            model_metrics["precisions"].append(metrics["precision_at_3"])
            model_metrics["top1s"].append(1 if metrics["top1_correct"] else 0)
            model_metrics["spreads"].append(metrics["spread"])

            top1_mark = "OK" if metrics["top1_correct"] else "MISS"
            print(f"  [{tc['id']:>7}] P@3={metrics['precision_at_3']:.2f}  "
                  f"Top1={top1_mark:<4}  "
                  f"Spread={metrics['spread']:.3f}  "
                  f"Latency={result['time_ms']:.0f}ms  "
                  f"Rank: {result['top_indices'][:3]}")

            model_metrics["details"].append({
                "id": tc["id"],
                "latency_ms": result["time_ms"],
                **metrics,
            })

        if model_metrics["latencies"]:
            all_results[model] = {
                "avg_latency_ms": statistics.mean(model_metrics["latencies"]),
                "p50_latency_ms": statistics.median(model_metrics["latencies"]),
                "avg_precision": statistics.mean(model_metrics["precisions"]),
                "top1_accuracy": statistics.mean(model_metrics["top1s"]),
                "avg_spread": statistics.mean(model_metrics["spreads"]),
                "details": model_metrics["details"],
            }

    # ── Summary table ──
    print(f"\n{'=' * 80}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 80}\n")

    header = f"{'Model':<25} {'P@3':>6} {'Top-1':>6} {'Spread':>8} {'Avg ms':>8} {'P50 ms':>8}"
    print(header)
    print("─" * len(header))

    for model, res in all_results.items():
        if "error" in res:
            print(f"{model:<25} {'ERROR':>6}")
            continue
        print(f"{model:<25} "
              f"{res['avg_precision']:>6.2f} "
              f"{res['top1_accuracy']:>6.2f} "
              f"{res['avg_spread']:>8.3f} "
              f"{res['avg_latency_ms']:>8.0f} "
              f"{res['p50_latency_ms']:>8.0f}")

    # ── Winner ──
    valid = {k: v for k, v in all_results.items() if "error" not in v}
    if len(valid) >= 2:
        best_quality = max(valid.items(), key=lambda x: x[1]["avg_precision"])
        fastest = min(valid.items(), key=lambda x: x[1]["avg_latency_ms"])
        print(f"\nBest quality:  {best_quality[0]} (P@3={best_quality[1]['avg_precision']:.2f})")
        print(f"Fastest:       {fastest[0]} ({fastest[1]['avg_latency_ms']:.0f}ms avg)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark reranker models")
    parser.add_argument("--url", default=DEFAULT_URL, help="API base URL")
    parser.add_argument("--model", help="Test single model only")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup call")
    args = parser.parse_args()

    models = [args.model] if args.model else MODELS

    run_benchmark(args.url, models, warmup=not args.no_warmup)


if __name__ == "__main__":
    main()
