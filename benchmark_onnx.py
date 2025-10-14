#!/usr/bin/env python3
"""
Benchmark ONNX INT8 Model Performance

Compares latency between:
- gemma-int8 (PyTorch quantized)
- gemma-onnx-int8 (ONNX INT8)

Goal: Verify 3-4x speedup with ONNX INT8
"""

import logging
import time
import numpy as np
from pathlib import Path
import torch
import torch.quantization as quant
import copy
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def measure_latency_pytorch(texts: List[str], num_runs: int = 10):
    """Measure latency for PyTorch INT8 gemma"""

    logger.info("\n" + "=" * 80)
    logger.info("⏱️  Benchmarking PyTorch INT8 (gemma-int8)")
    logger.info("=" * 80)

    try:
        from sentence_transformers import SentenceTransformer

        # Load and quantize model
        logger.info("Loading SentenceTransformer model...")
        model_base = SentenceTransformer("google/embeddinggemma-300m")
        model_base = model_base.to("cpu")

        logger.info("Applying INT8 quantization...")
        model_int8 = copy.deepcopy(model_base)
        model_int8 = model_int8.to("cpu")
        model_int8 = quant.quantize_dynamic(
            model_int8,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        logger.info("✅ Model loaded and quantized")

        # Warmup
        logger.info("\nWarming up...")
        _ = model_int8.encode(texts[:1], show_progress_bar=False, normalize_embeddings=True)

        # Benchmark
        logger.info(f"Running {num_runs} iterations...")
        latencies = []

        for i in range(num_runs):
            start = time.time()
            _ = model_int8.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=8,
                convert_to_numpy=True
            )
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)

            if (i + 1) % 5 == 0:
                logger.info(f"  Run {i+1}/{num_runs}: {latency:.2f}ms")

        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        per_text_latency = avg_latency / len(texts)

        logger.info(f"\n📊 PyTorch INT8 Results:")
        logger.info(f"   Avg latency: {avg_latency:.2f}ms ± {std_latency:.2f}ms")
        logger.info(f"   Per text: {per_text_latency:.2f}ms")

        return {
            "avg_ms": avg_latency,
            "std_ms": std_latency,
            "per_text_ms": per_text_latency
        }

    except Exception as e:
        logger.error(f"❌ PyTorch benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def measure_latency_onnx(texts: List[str], num_runs: int = 10):
    """Measure latency for ONNX INT8 gemma"""

    logger.info("\n" + "=" * 80)
    logger.info("⏱️  Benchmarking ONNX INT8 (gemma-onnx-int8)")
    logger.info("=" * 80)

    model_path = Path("./models/gemma-onnx-int8")

    if not model_path.exists():
        logger.error(f"❌ ONNX model not found: {model_path}")
        logger.error("Please run convert_to_onnx.py first")
        return None

    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        logger.info(f"Loading ONNX model from {model_path}...")
        model = ORTModelForFeatureExtraction.from_pretrained(
            str(model_path),
            provider="CPUExecutionProvider"
        )

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        logger.info("✅ ONNX model loaded")

        # Warmup
        logger.info("\nWarming up...")
        inputs = tokenizer(texts[:1], padding=True, truncation=True, return_tensors="pt")
        _ = model(**inputs)

        # Benchmark
        logger.info(f"Running {num_runs} iterations...")
        latencies = []

        for i in range(num_runs):
            start = time.time()

            # Tokenize
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

            # Generate embeddings
            outputs = model(**inputs)

            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()

            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)

            if (i + 1) % 5 == 0:
                logger.info(f"  Run {i+1}/{num_runs}: {latency:.2f}ms")

        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        per_text_latency = avg_latency / len(texts)

        logger.info(f"\n📊 ONNX INT8 Results:")
        logger.info(f"   Avg latency: {avg_latency:.2f}ms ± {std_latency:.2f}ms")
        logger.info(f"   Per text: {per_text_latency:.2f}ms")

        return {
            "avg_ms": avg_latency,
            "std_ms": std_latency,
            "per_text_ms": per_text_latency
        }

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Please install: pip install onnxruntime optimum[onnxruntime]")
        return None
    except Exception as e:
        logger.error(f"❌ ONNX benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main benchmark execution"""

    logger.info("=" * 80)
    logger.info("🔬 ONNX INT8 Performance Benchmark")
    logger.info("=" * 80)

    # Test texts
    test_texts = [
        "Machine learning is a subset of artificial intelligence",
        "Neural networks are inspired by biological neurons",
        "Deep learning uses multiple layers of neural networks",
        "Natural language processing enables computers to understand text",
        "Embeddings represent text as dense vectors in high-dimensional space"
    ]

    logger.info(f"\nTest configuration:")
    logger.info(f"  - Texts: {len(test_texts)}")
    logger.info(f"  - Runs: 10")
    logger.info(f"  - Device: CPU")

    # Benchmark PyTorch INT8
    pytorch_results = measure_latency_pytorch(test_texts, num_runs=10)

    # Benchmark ONNX INT8
    onnx_results = measure_latency_onnx(test_texts, num_runs=10)

    # Compare results
    if pytorch_results and onnx_results:
        logger.info("\n" + "=" * 80)
        logger.info("📊 COMPARISON SUMMARY")
        logger.info("=" * 80)

        pytorch_latency = pytorch_results["avg_ms"]
        onnx_latency = onnx_results["avg_ms"]
        speedup = pytorch_latency / onnx_latency

        logger.info(f"\nLatency (batch of {len(test_texts)} texts):")
        logger.info(f"  PyTorch INT8:  {pytorch_latency:.2f}ms")
        logger.info(f"  ONNX INT8:     {onnx_latency:.2f}ms")
        logger.info(f"  Speedup:       {speedup:.2f}x")

        logger.info(f"\nPer-text latency:")
        logger.info(f"  PyTorch INT8:  {pytorch_results['per_text_ms']:.2f}ms")
        logger.info(f"  ONNX INT8:     {onnx_results['per_text_ms']:.2f}ms")

        logger.info("\n" + "-" * 80)
        logger.info("💡 Analysis:")

        if speedup >= 3.0:
            logger.info(f"  ✅ EXCELLENT: {speedup:.1f}x speedup achieved!")
            logger.info(f"  ✅ Target of 3-4x speedup MET")
        elif speedup >= 2.0:
            logger.info(f"  ⚖️  GOOD: {speedup:.1f}x speedup achieved")
            logger.info(f"  ⚠️  Target of 3-4x speedup not fully met, but still useful")
        else:
            logger.info(f"  ⚠️  MODERATE: {speedup:.1f}x speedup achieved")
            logger.info(f"  ❌ Target of 3-4x speedup NOT met")

        # Railway production estimate
        logger.info(f"\n🚀 Railway Production Estimate (2048 tokens):")
        logger.info(f"   Assuming ~2x longer for 2048 tokens:")
        pytorch_prod = pytorch_results['per_text_ms'] * 2
        onnx_prod = onnx_results['per_text_ms'] * 2
        logger.info(f"   PyTorch INT8:  ~{pytorch_prod:.1f}ms per embedding")
        logger.info(f"   ONNX INT8:     ~{onnx_prod:.1f}ms per embedding")

        if onnx_prod < 15:
            logger.info(f"   ✅ ONNX meets Railway target (<15ms)")
        elif onnx_prod < 20:
            logger.info(f"   ⚖️  ONNX close to Railway target")
        else:
            logger.info(f"   ⚠️  ONNX may still be slow for Railway")

        logger.info("\n" + "=" * 80)

    elif not pytorch_results:
        logger.error("\n❌ PyTorch benchmark failed - cannot compare")
    elif not onnx_results:
        logger.error("\n❌ ONNX benchmark failed - cannot compare")

    logger.info("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
