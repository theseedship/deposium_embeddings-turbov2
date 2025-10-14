#!/usr/bin/env python3
"""
LEAF Model Comparison: v1 (512 tokens) vs v2 (2048 tokens)

Compares MTEB performance, speed, and quality metrics between versions.
Generates comparison tables and visualizations.

Usage:
    python compare_versions.py --v1-model models/v1/model_quantized.pt \
                               --v2-model models/v2/model_quantized.pt \
                               --output comparison_report.md
"""

import torch
import numpy as np
from transformers import AutoTokenizer
from pathlib import Path
import time
import argparse
import json
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL WRAPPER
# ============================================================================
class LEAFModelWrapper:
    """Wrapper for LEAF model with MTEB compatibility"""

    def __init__(self, model_path: str, version: str = "v1"):
        self.version = version
        self.model_path = model_path
        logger.info(f"Loading {version} model from {model_path}")

        # Load model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model = checkpoint['model']
        self.model.eval()

        # Load tokenizer
        tokenizer_path = Path(model_path).parent
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        self.model.set_tokenizer(self.tokenizer)

        # Get model info
        self.max_seq_length = checkpoint.get('max_seq_length', 512 if version == "v1" else 2048)
        self.num_params = sum(p.numel() for p in self.model.parameters())

        logger.info(f"✅ {version} model loaded: {self.num_params/1e6:.1f}M params, {self.max_seq_length} tokens")

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode sentences to embeddings (MTEB-compatible interface)"""
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            with torch.no_grad():
                embeddings = self.model.encode(
                    batch,
                    device='cpu',
                    normalize=normalize_embeddings
                )

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


# ============================================================================
# BENCHMARKING
# ============================================================================
def benchmark_speed(model: LEAFModelWrapper, num_samples: int = 1000) -> Dict[str, float]:
    """Benchmark inference speed"""
    logger.info(f"Benchmarking {model.version} speed with {num_samples} samples...")

    # Generate test data
    texts = [f"This is a test sentence number {i} for speed benchmarking." for i in range(num_samples)]

    # Warmup
    _ = model.encode(texts[:10], batch_size=32)

    # Benchmark
    start_time = time.time()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    end_time = time.time()

    elapsed = end_time - start_time
    throughput = num_samples / elapsed
    latency = (elapsed / num_samples) * 1000  # ms

    results = {
        "throughput_texts_per_sec": throughput,
        "latency_ms_per_text": latency,
        "total_time_sec": elapsed,
        "embedding_dims": embeddings.shape[1],
    }

    logger.info(f"  Throughput: {throughput:.1f} texts/s")
    logger.info(f"  Latency: {latency:.2f} ms/text")

    return results


def benchmark_quality_mteb(model: LEAFModelWrapper, tasks: List[str] = None) -> Dict[str, Any]:
    """Run MTEB evaluation"""
    if tasks is None:
        tasks = ["STSBenchmark"]  # Quick task for comparison

    logger.info(f"Running MTEB evaluation on {model.version}...")

    try:
        from mteb import MTEB
    except ImportError:
        logger.error("MTEB not installed. Install with: pip install mteb")
        return {}

    evaluation = MTEB(tasks=tasks)
    results = evaluation.run(
        model,
        output_folder=f"comparison_results/{model.version}",
        eval_splits=["test"]
    )

    return results


# ============================================================================
# COMPARISON REPORT
# ============================================================================
def generate_comparison_report(
    v1_results: Dict[str, Any],
    v2_results: Dict[str, Any],
    output_path: str = "COMPARISON_REPORT.md"
):
    """Generate comprehensive comparison report"""
    logger.info(f"Generating comparison report: {output_path}")

    report = []

    # Header
    report.append("# LEAF Model Comparison: v1 vs v2\n")
    report.append("**Comparison Date**: 2025-10-12\n")
    report.append("**Evaluation**: MTEB + Speed Benchmarks\n")
    report.append("\n---\n")

    # Architecture Comparison
    report.append("## 📐 Architecture Comparison\n")
    report.append("| Property | v1 (512 tokens) | v2 (2048 tokens) | Change |\n")
    report.append("|----------|-----------------|------------------|--------|\n")
    report.append(f"| **Layers** | 6 | 12 | +100% ✅ |\n")
    report.append(f"| **Parameters** | 75M | 120M | +60% ✅ |\n")
    report.append(f"| **Context Length** | 512 | 2048 | +300% ✅ |\n")
    report.append(f"| **Hidden Size Ratio** | 0.5x | 0.75x | +50% ✅ |\n")
    report.append(f"| **Training Data** | 50k samples | 200k samples | +300% ✅ |\n")
    report.append(f"| **Training Epochs** | 3 | 10 | +233% ✅ |\n")
    report.append(f"| **Alignment Loss Weight** | 1.0 | 2.5 | +150% ✅ |\n")
    report.append("\n")

    # Speed Comparison
    report.append("## ⚡ Speed Comparison\n")
    report.append("| Metric | v1 | v2 | Change |\n")
    report.append("|--------|----|----|--------|\n")

    v1_speed = v1_results.get("speed", {})
    v2_speed = v2_results.get("speed", {})

    v1_throughput = v1_speed.get("throughput_texts_per_sec", 0)
    v2_throughput = v2_speed.get("throughput_texts_per_sec", 0)
    throughput_change = ((v2_throughput - v1_throughput) / v1_throughput * 100) if v1_throughput > 0 else 0

    v1_latency = v1_speed.get("latency_ms_per_text", 0)
    v2_latency = v2_speed.get("latency_ms_per_text", 0)
    latency_change = ((v2_latency - v1_latency) / v1_latency * 100) if v1_latency > 0 else 0

    report.append(f"| **Throughput** | {v1_throughput:.1f} texts/s | {v2_throughput:.1f} texts/s | {throughput_change:+.1f}% |\n")
    report.append(f"| **Latency** | {v1_latency:.2f} ms | {v2_latency:.2f} ms | {latency_change:+.1f}% |\n")
    report.append(f"| **Embedding Dims** | {v1_speed.get('embedding_dims', 768)} | {v2_speed.get('embedding_dims', 768)} | Same |\n")
    report.append("\n")

    # Quality Comparison (MTEB)
    report.append("## 📊 Quality Comparison (MTEB)\n")

    # Check if we have MTEB results
    v1_mteb = v1_results.get("mteb", {})
    v2_mteb = v2_results.get("mteb", {})

    if v1_mteb or v2_mteb:
        report.append("| Task | Metric | v1 | v2 | Improvement |\n")
        report.append("|------|--------|----|----|-------------|\n")

        # STSBenchmark
        v1_sts = v1_mteb.get("STSBenchmark", {}).get("test", {}).get("spearman", 0.223)
        v2_sts = v2_mteb.get("STSBenchmark", {}).get("test", {}).get("spearman", 0)
        sts_improvement = ((v2_sts - v1_sts) / v1_sts * 100) if v1_sts > 0 else 0

        report.append(f"| **STSBenchmark** | Spearman | {v1_sts:.3f} | {v2_sts:.3f} | {sts_improvement:+.1f}% |\n")

    else:
        # Use known v1 results
        report.append("### Known Results (v1)\n\n")
        report.append("| Task | Metric | v1 (FAILED) | v2 (Target) | Target Improvement |\n")
        report.append("|------|--------|-------------|-------------| -------------------|\n")
        report.append(f"| **STSBenchmark** | Spearman | 0.223 | 0.70+ | **+214%** 🎯 |\n")
        report.append(f"| **STS22 English** | Spearman | 0.373 | 0.65+ | **+74%** 🎯 |\n")
        report.append(f"| **STS22 Average** | Spearman | ~0.21 | 0.50+ | **+138%** 🎯 |\n")
        report.append(f"| **Cross-lingual** | Spearman | -0.14 | 0.30+ | **Complete Fix** 🎯 |\n")
        report.append(f"| **MTEB Score (est.)** | Overall | ~25 | 55+ | **+120%** 🎯 |\n")

    report.append("\n")

    # Detailed v1 Results (Known)
    report.append("## ❌ Detailed v1 Results (FAILED)\n\n")
    report.append("### STSBenchmark\n")
    report.append("- **Spearman**: 0.223 (Target: 0.81)\n")
    report.append("- **Quality Loss**: -72% vs base model\n")
    report.append("- **Status**: ❌ CRITICAL FAILURE\n\n")

    report.append("### STS22 by Language\n")
    report.append("| Language | Spearman | Status |\n")
    report.append("|----------|----------|--------|\n")
    report.append("| 🇨🇳 Chinese | 0.499 | 🟡 Best (still poor) |\n")
    report.append("| 🇸🇦 Arabic | 0.469 | 🟡 Moderate |\n")
    report.append("| 🇮🇹 Italian | 0.435 | 🟡 Moderate |\n")
    report.append("| 🇪🇸 Spanish | 0.403 | 🟠 Poor |\n")
    report.append("| 🇬🇧 English | 0.373 | 🟠 Poor |\n")
    report.append("| 🇫🇷 French | 0.300 | 🔴 Very poor |\n")
    report.append("| 🇷🇺 Russian | 0.268 | 🔴 Very poor |\n")
    report.append("| 🇹🇷 Turkish | 0.247 | 🔴 Very poor |\n")
    report.append("| 🇩🇪 German | 0.163 | ❌ Critical |\n")
    report.append("| 🇵🇱 Polish | 0.132 | ❌ Critical |\n\n")

    report.append("### Cross-lingual (Translation Tasks)\n")
    report.append("| Pair | Spearman | Status |\n")
    report.append("|------|----------|--------|\n")
    report.append("| 🇪🇸-🇮🇹 | 0.119 | ❌ Failed |\n")
    report.append("| 🇩🇪-🇵🇱 | 0.113 | ❌ Failed |\n")
    report.append("| 🇩🇪-🇫🇷 | 0.070 | ❌ Failed |\n")
    report.append("| 🇪🇸-🇬🇧 | 0.002 | ❌ Random |\n")
    report.append("| 🇨🇳-🇬🇧 | -0.012 | ❌ Inverse |\n")
    report.append("| 🇵🇱-🇬🇧 | -0.143 | ❌ **WORST** |\n\n")

    # Summary
    report.append("## 📝 Summary\n\n")
    report.append("### v1 (512 tokens) - FAILED\n")
    report.append("- ❌ **Architecture too aggressive**: 6 layers insufficient\n")
    report.append("- ❌ **Data insufficient**: 50k samples, mostly English\n")
    report.append("- ❌ **High alignment loss**: 2.18 (warning sign)\n")
    report.append("- ❌ **Quality catastrophic**: -72% vs base model\n")
    report.append("- ❌ **Multilingual destroyed**: Cross-lingual scores negative\n")
    report.append("- ✅ **Speed excellent**: 695 texts/s\n\n")

    report.append("### v2 (2048 tokens) - Expected Improvements\n")
    report.append("- ✅ **Architecture doubled**: 12 layers (2x)\n")
    report.append("- ✅ **Data 4x larger**: 200k multilingual samples\n")
    report.append("- ✅ **Alignment prioritized**: Weight 2.5 (vs 1.0)\n")
    report.append("- ✅ **Curriculum learning**: 512→1024→2048 progressive\n")
    report.append("- ✅ **Quality monitoring**: MTEB validation every 1000 steps\n")
    report.append("- ✅ **Target**: MTEB 55+ (vs ~25 in v1)\n")
    report.append("- ⚠️ **Speed trade-off**: Likely ~400-500 texts/s (still fast)\n\n")

    # Recommendations
    report.append("## 🎯 Recommendations\n\n")
    report.append("1. **Proceed with v2 training** using the improved configuration\n")
    report.append("2. **Monitor alignment loss** - stop if > 1.5 after epoch 3\n")
    report.append("3. **Validate frequently** - MTEB STSBenchmark every 1000 steps\n")
    report.append("4. **Target quality** - Spearman 0.70+ on STSBenchmark\n")
    report.append("5. **Expected training time** - 12-15 hours on RTX 4050\n\n")

    # Write report
    report_text = "".join(report)
    Path(output_path).write_text(report_text)
    logger.info(f"✅ Comparison report saved to {output_path}")

    return report_text


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Compare LEAF v1 vs v2 models")
    parser.add_argument("--v1-model", type=str, help="Path to v1 model")
    parser.add_argument("--v2-model", type=str, help="Path to v2 model (optional)")
    parser.add_argument("--output", type=str, default="COMPARISON_REPORT.md", help="Output report path")
    parser.add_argument("--benchmark-speed", action="store_true", help="Run speed benchmarks")
    parser.add_argument("--benchmark-mteb", action="store_true", help="Run MTEB benchmarks (slow)")
    args = parser.parse_args()

    results = {
        "v1": {},
        "v2": {}
    }

    # Load v1 model (if provided)
    if args.v1_model and Path(args.v1_model).exists():
        v1_model = LEAFModelWrapper(args.v1_model, version="v1")

        if args.benchmark_speed:
            results["v1"]["speed"] = benchmark_speed(v1_model)

        if args.benchmark_mteb:
            results["v1"]["mteb"] = benchmark_quality_mteb(v1_model)

    # Load v2 model (if provided)
    if args.v2_model and Path(args.v2_model).exists():
        v2_model = LEAFModelWrapper(args.v2_model, version="v2")

        if args.benchmark_speed:
            results["v2"]["speed"] = benchmark_speed(v2_model)

        if args.benchmark_mteb:
            results["v2"]["mteb"] = benchmark_quality_mteb(v2_model)

    # Generate report
    report = generate_comparison_report(results["v1"], results["v2"], args.output)

    logger.info("✅ Comparison complete!")
    logger.info(f"📄 Report: {args.output}")


if __name__ == "__main__":
    main()
