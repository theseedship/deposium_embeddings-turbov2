#!/usr/bin/env python3
"""
MTEB Comparison: gemma-int8 vs TurboX.v2 (Model2Vec)

Compares:
- Quality: MTEB scores across multiple tasks
- Speed: Inference latency
- Size: Model footprint

Use cases:
- Determines if gemma should be distilled to Model2Vec
- Informs Model2Vec distillation decision
"""

import torch
import torch.quantization as quant
import copy
import time
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from model2vec import StaticModel
from pathlib import Path
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelWrapper:
    """Unified wrapper for MTEB compatibility"""

    def __init__(self, model, model_name: str, model_type: str):
        self.model = model
        self.model_name = model_name
        self.model_type = model_type  # "gemma-int8" or "turbov2"

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True,
        **kwargs
    ) -> np.ndarray:
        """MTEB-compatible encoding interface"""

        if self.model_type == "gemma-int8":
            # EmbeddingGemma INT8 variant
            embeddings = self.model.encode(
                sentences,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
                batch_size=batch_size,
                convert_to_numpy=True
            )
            # Ensure float32 for compatibility
            embeddings = np.array(embeddings, dtype=np.float32)

        elif self.model_type == "turbov2":
            # Model2Vec (TurboX.v2)
            embeddings = self.model.encode(
                sentences,
                show_progress_bar=show_progress_bar
            )
            # Convert to numpy if needed
            if isinstance(embeddings, list):
                embeddings = np.array([emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings])
            elif not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

        return embeddings


def load_models():
    """Load both models for comparison"""
    models = {}

    logger.info("=" * 60)
    logger.info("🔧 Loading Models")
    logger.info("=" * 60)

    # Load gemma-int8 (INT8 quantized)
    try:
        logger.info("\n📦 Loading gemma-int8 (INT8 quantized)...")

        # Load base model in float32
        model_base = SentenceTransformer("google/embeddinggemma-300m")
        model_base = model_base.to("cpu")

        # Create INT8 quantized copy
        model_int8 = copy.deepcopy(model_base)
        model_int8 = model_int8.to("cpu")

        # Apply INT8 quantization
        model_int8 = quant.quantize_dynamic(
            model_int8,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        models["gemma-int8"] = ModelWrapper(model_int8, "gemma-int8", "gemma-int8")
        logger.info("✅ gemma-int8 loaded (~300MB, INT8)")

    except Exception as e:
        logger.error(f"❌ Failed to load gemma-int8: {e}")
        return None

    # Load TurboX.v2 (Model2Vec)
    try:
        logger.info("\n📦 Loading TurboX.v2 (Model2Vec)...")
        model_turbov2 = StaticModel.from_pretrained("C10X/Qwen3-Embedding-TurboX.v2")
        models["turbov2"] = ModelWrapper(model_turbov2, "turbov2", "turbov2")
        logger.info("✅ TurboX.v2 loaded (~30MB, static embeddings)")

    except Exception as e:
        logger.error(f"❌ Failed to load TurboX.v2: {e}")
        return None

    logger.info("\n✅ All models loaded successfully!\n")
    return models


def measure_latency(model_wrapper: ModelWrapper, test_texts: List[str], num_runs: int = 10):
    """Measure inference latency"""
    logger.info(f"⏱️  Measuring latency for {model_wrapper.model_name}...")

    latencies = []

    # Warmup
    _ = model_wrapper.encode(test_texts[:1], show_progress_bar=False)

    # Measure
    for _ in range(num_runs):
        start = time.time()
        _ = model_wrapper.encode(test_texts, show_progress_bar=False)
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)

    logger.info(f"   Avg: {avg_latency:.2f}ms ± {std_latency:.2f}ms")

    return {
        "avg_ms": avg_latency,
        "std_ms": std_latency,
        "per_text_ms": avg_latency / len(test_texts)
    }


def run_mteb_evaluation(models: Dict[str, ModelWrapper]):
    """Run MTEB benchmarks on selected tasks"""

    try:
        from mteb import MTEB
        logger.info("✅ MTEB imported successfully")
    except ImportError:
        logger.error("❌ MTEB not installed. Install with: pip install mteb")
        return None

    # Select fast, representative tasks
    # Focus on STS (Semantic Textual Similarity) which is core for embeddings
    tasks = [
        "STSBenchmark",  # Standard STS benchmark
        "STS17",         # Cross-lingual STS
        "BIOSSES",       # Domain-specific (biomedical)
    ]

    logger.info("=" * 60)
    logger.info("📊 Running MTEB Evaluation")
    logger.info("=" * 60)
    logger.info(f"Tasks: {tasks}")
    logger.info("This may take 10-20 minutes...\n")

    results = {}

    for model_name, model_wrapper in models.items():
        logger.info(f"\n🎯 Evaluating {model_name}...")
        logger.info("-" * 60)

        evaluation = MTEB(tasks=tasks)

        try:
            model_results = evaluation.run(
                model_wrapper,
                output_folder=f"mteb_results/{model_name}",
                eval_splits=["test"],
                overwrite_results=True
            )
            results[model_name] = model_results
            logger.info(f"✅ {model_name} evaluation complete!")

        except Exception as e:
            logger.error(f"❌ Failed to evaluate {model_name}: {e}")
            results[model_name] = None

    return results


def extract_scores(results: Dict) -> Dict:
    """Extract key metrics from MTEB results"""

    scores = {}

    for model_name, model_results in results.items():
        if model_results is None:
            scores[model_name] = None
            continue

        model_scores = {}

        for task_name, task_results in model_results.items():
            if "test" in task_results:
                test_results = task_results["test"]

                # Extract main metric (Spearman for STS tasks)
                if "cos_sim" in test_results:
                    cos_sim = test_results["cos_sim"]
                    if "spearman" in cos_sim:
                        model_scores[task_name] = cos_sim["spearman"]
                    elif "pearson" in cos_sim:
                        model_scores[task_name] = cos_sim["pearson"]
                elif "main_score" in test_results:
                    model_scores[task_name] = test_results["main_score"]

        scores[model_name] = model_scores

    return scores


def print_comparison(scores: Dict, latencies: Dict, model_sizes: Dict):
    """Print side-by-side comparison"""

    logger.info("\n" + "=" * 80)
    logger.info("📊 COMPARISON RESULTS: gemma-int8 vs TurboX.v2")
    logger.info("=" * 80)

    # Model Info
    logger.info("\n🔧 Model Information:")
    logger.info("-" * 80)
    logger.info(f"{'Model':<20} {'Type':<20} {'Size':<15} {'Dimensions':<15}")
    logger.info("-" * 80)
    logger.info(f"{'gemma-int8':<20} {'INT8 Quantized':<20} {'~300MB':<15} {'768D':<15}")
    logger.info(f"{'TurboX.v2':<20} {'Model2Vec (Static)':<20} {'~30MB':<15} {'1024D':<15}")

    # Latency Comparison
    logger.info("\n⏱️  Inference Latency (10 runs, batch of 5 texts):")
    logger.info("-" * 80)
    logger.info(f"{'Model':<20} {'Avg (ms)':<15} {'Per Text (ms)':<20} {'Speedup':<15}")
    logger.info("-" * 80)

    gemma_latency = latencies["gemma-int8"]["avg_ms"]
    turbov2_latency = latencies["turbov2"]["avg_ms"]
    speedup = gemma_latency / turbov2_latency

    logger.info(f"{'gemma-int8':<20} {gemma_latency:<15.2f} {latencies['gemma-int8']['per_text_ms']:<20.2f} {'1.0x':<15}")
    logger.info(f"{'TurboX.v2':<20} {turbov2_latency:<15.2f} {latencies['turbov2']['per_text_ms']:<20.2f} {f'{speedup:.1f}x faster':<15}")

    # Quality Comparison (MTEB Scores)
    logger.info("\n🎯 MTEB Quality Scores (Spearman Correlation):")
    logger.info("-" * 80)

    if scores["gemma-int8"] and scores["turbov2"]:
        # Header
        logger.info(f"{'Task':<25} {'gemma-int8':<20} {'TurboX.v2':<20} {'Difference':<15}")
        logger.info("-" * 80)

        # Task scores
        all_tasks = set(scores["gemma-int8"].keys()) | set(scores["turbov2"].keys())
        gemma_avg = []
        turbo_avg = []

        for task in sorted(all_tasks):
            gemma_score = scores["gemma-int8"].get(task, 0)
            turbo_score = scores["turbov2"].get(task, 0)
            diff = gemma_score - turbo_score

            if gemma_score > 0:
                gemma_avg.append(gemma_score)
            if turbo_score > 0:
                turbo_avg.append(turbo_score)

            logger.info(f"{task:<25} {gemma_score:<20.4f} {turbo_score:<20.4f} {diff:+.4f}")

        # Average
        logger.info("-" * 80)
        gemma_mean = np.mean(gemma_avg) if gemma_avg else 0
        turbo_mean = np.mean(turbo_avg) if turbo_avg else 0
        diff_mean = gemma_mean - turbo_mean

        logger.info(f"{'AVERAGE':<25} {gemma_mean:<20.4f} {turbo_mean:<20.4f} {diff_mean:+.4f}")

    # Decision Guidance
    logger.info("\n" + "=" * 80)
    logger.info("💡 RECOMMENDATION")
    logger.info("=" * 80)

    if scores["gemma-int8"] and scores["turbov2"]:
        quality_loss = (gemma_mean - turbo_mean) * 100  # percentage points

        logger.info(f"\nQuality: gemma-int8 is {quality_loss:+.1f} percentage points {'better' if quality_loss > 0 else 'worse'}")
        logger.info(f"Speed: TurboX.v2 is {speedup:.1f}x faster")
        logger.info(f"Size: TurboX.v2 is 10x smaller (~30MB vs ~300MB)")

        logger.info("\n📋 Analysis:")

        if quality_loss > 10:
            logger.info("   ✅ gemma-int8 has significantly better quality (+10+ points)")
            logger.info("   ⚠️  Consider Model2Vec distillation ONLY if speed is critical")
            logger.info("   💡 Recommendation: Keep gemma-int8 for production")
        elif quality_loss > 5:
            logger.info("   ⚖️  gemma-int8 has moderately better quality (+5-10 points)")
            logger.info("   💡 Consider use case: high-quality → gemma-int8, high-volume → TurboX.v2")
        elif quality_loss > -5:
            logger.info("   ⚖️  Models have similar quality (±5 points)")
            logger.info("   💡 TurboX.v2 is better for Railway CPU (faster, smaller)")
        else:
            logger.info("   ❌ gemma-int8 has worse quality (-5+ points)")
            logger.info("   💡 TurboX.v2 is clearly better (faster AND better quality)")

        logger.info("\n🚀 Model2Vec Distillation Decision:")
        if quality_loss > 10:
            logger.info("   ❌ NOT RECOMMENDED - Quality loss would be too high")
        elif quality_loss > 5:
            logger.info("   ⚠️  EVALUATE - Test on your specific use case first")
        else:
            logger.info("   ✅ RECOMMENDED - TurboX.v2 is already better or comparable")

    logger.info("\n" + "=" * 80)


def save_results(scores: Dict, latencies: Dict, output_file: str = "mteb_comparison_results.json"):
    """Save results to JSON for future reference"""

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "gemma-int8": {
                "type": "INT8 Quantized",
                "size_mb": 300,
                "dimensions": 768,
                "mteb_scores": scores.get("gemma-int8", {}),
                "latency": latencies.get("gemma-int8", {})
            },
            "turbov2": {
                "type": "Model2Vec (Static)",
                "size_mb": 30,
                "dimensions": 1024,
                "mteb_scores": scores.get("turbov2", {}),
                "latency": latencies.get("turbov2", {})
            }
        }
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n💾 Results saved to: {output_file}")


def main():
    """Main execution"""

    logger.info("🔬 MTEB Comparison: gemma-int8 vs TurboX.v2")
    logger.info("=" * 80)
    logger.info("This benchmark will:")
    logger.info("  1. Load both models")
    logger.info("  2. Measure inference latency")
    logger.info("  3. Run MTEB quality benchmarks")
    logger.info("  4. Compare results and provide recommendations")
    logger.info("=" * 80)
    logger.info("")

    # Load models
    models = load_models()
    if not models:
        logger.error("❌ Failed to load models. Exiting.")
        return

    # Test texts for latency measurement
    test_texts = [
        "Machine learning is a subset of artificial intelligence",
        "Neural networks are inspired by biological neurons",
        "Deep learning uses multiple layers of neural networks",
        "Natural language processing enables computers to understand text",
        "Embeddings represent text as dense vectors in high-dimensional space"
    ]

    # Measure latency
    logger.info("=" * 60)
    logger.info("⏱️  Latency Benchmarks")
    logger.info("=" * 60)

    latencies = {}
    for model_name, model_wrapper in models.items():
        latencies[model_name] = measure_latency(model_wrapper, test_texts, num_runs=10)

    # Run MTEB evaluation
    mteb_results = run_mteb_evaluation(models)

    if mteb_results is None:
        logger.error("❌ MTEB evaluation failed. Exiting.")
        return

    # Extract scores
    scores = extract_scores(mteb_results)

    # Model sizes (approximations)
    model_sizes = {
        "gemma-int8": 300,  # MB
        "turbov2": 30       # MB
    }

    # Print comparison
    print_comparison(scores, latencies, model_sizes)

    # Save results
    save_results(scores, latencies)

    logger.info("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
