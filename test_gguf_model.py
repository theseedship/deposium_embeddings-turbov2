#!/usr/bin/env python3
"""
Test GGUF Q4_K_M Model Loading

Verifies that the GGUF model loads correctly with llama-cpp-python
and can generate embeddings.
"""

import logging
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_gguf_model():
    """Test GGUF model loading and embedding generation"""

    logger.info("=" * 80)
    logger.info("🔬 Testing GGUF Q4_K_M Model")
    logger.info("=" * 80)

    # Import llama-cpp-python
    try:
        from llama_cpp import Llama
        logger.info("✅ llama-cpp-python imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import llama-cpp-python: {e}")
        logger.error("Please install: pip install llama-cpp-python")
        return False

    # Find GGUF model file
    model_dir = Path("./models/gemma-gguf-q4")
    gguf_files = list(model_dir.glob("*.gguf"))

    if not gguf_files:
        logger.error(f"❌ No GGUF files found in {model_dir}")
        return False

    model_path = gguf_files[0]
    logger.info(f"\n📁 Model path: {model_path}")
    logger.info(f"📊 Model size: {model_path.stat().st_size / (1024*1024):.1f} MB")

    try:
        logger.info("\n⏳ Loading GGUF model...")
        logger.info("(This may take a few seconds on first load)")

        # Load model with embedding mode
        # Note: embedding=True enables embedding generation
        llm = Llama(
            model_path=str(model_path),
            embedding=True,
            n_ctx=2048,  # Context window
            n_threads=None,  # Use all available CPU threads
            verbose=False
        )

        logger.info("✅ GGUF model loaded successfully")

    except Exception as e:
        logger.error(f"❌ Failed to load GGUF model: {e}")
        return False

    # Test embedding generation
    logger.info("\n🧪 Testing embedding generation...")

    test_texts = [
        "This is a test sentence",
        "Machine learning is fascinating",
    ]

    try:
        embeddings = []

        for i, text in enumerate(test_texts, 1):
            logger.info(f"  Processing text {i}/{len(test_texts)}: {text[:50]}...")

            # Generate embedding
            # Note: llama-cpp-python returns embeddings as a list
            embedding = llm.embed(text)

            embeddings.append(embedding)
            logger.info(f"    ✓ Generated embedding: {len(embedding)}D")

        logger.info("\n✅ Embedding generation successful")

        # Validate embeddings
        logger.info("\n🔍 Validating embeddings...")

        for i, emb in enumerate(embeddings, 1):
            emb_array = np.array(emb)

            # Check for NaN/Inf
            has_nan = np.isnan(emb_array).any()
            has_inf = np.isinf(emb_array).any()
            is_zero = np.allclose(emb_array, 0)

            logger.info(f"  Embedding {i}:")
            logger.info(f"    Shape: {emb_array.shape}")
            logger.info(f"    Has NaN: {has_nan}")
            logger.info(f"    Has Inf: {has_inf}")
            logger.info(f"    Is zero: {is_zero}")
            logger.info(f"    Mean: {emb_array.mean():.6f}")
            logger.info(f"    Std: {emb_array.std():.6f}")

            if has_nan or has_inf or is_zero:
                logger.error("    ❌ Invalid embedding detected!")
                return False

        # Compute cosine similarity
        logger.info("\n📐 Computing cosine similarity...")

        emb1 = np.array(embeddings[0])
        emb2 = np.array(embeddings[1])

        # Normalize
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)

        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)

        logger.info(f"  Cosine similarity: {similarity:.4f}")

        if 0.0 <= similarity <= 1.0:
            logger.info("  ✅ Similarity in valid range")
        else:
            logger.warning(f"  ⚠️  Unusual similarity value: {similarity}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 80)

        logger.info("\n📊 Model Info:")
        logger.info(f"  Embedding dimension: {len(embeddings[0])}D")
        logger.info(f"  Model size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        logger.info(f"  Quantization: Q4_K_M (4-bit)")

        logger.info("\n🚀 Next steps:")
        logger.info("  1. Benchmark performance: python3 benchmark_gguf.py")
        logger.info("  2. Compare with ONNX and PyTorch INT8")
        logger.info("  3. Run MTEB evaluation if performance is good")

        return True

    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main execution"""
    import sys

    success = test_gguf_model()

    if success:
        logger.info("\n🎉 GGUF model test successful!")
        sys.exit(0)
    else:
        logger.error("\n❌ GGUF model test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
