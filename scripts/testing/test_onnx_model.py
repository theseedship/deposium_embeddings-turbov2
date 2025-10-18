#!/usr/bin/env python3
"""
Test ONNX Model Loading and Basic Functionality

Verifies that the ONNX INT8 model can be loaded and generates valid embeddings.
"""

import logging
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_onnx_model():
    """Test ONNX model loading and embedding generation"""

    model_path = Path("./models/gemma-onnx-int8")

    logger.info("=" * 80)
    logger.info("🧪 Testing ONNX INT8 Model")
    logger.info("=" * 80)

    # Check if model exists
    if not model_path.exists():
        logger.error(f"❌ Model directory not found: {model_path}")
        logger.error("Please run convert_to_onnx.py first")
        return False

    # Check for ONNX files (may be in onnx/ subdirectory)
    onnx_files = list(model_path.glob("*.onnx")) + list(model_path.glob("**/*.onnx"))
    if not onnx_files:
        logger.error(f"❌ No ONNX files found in {model_path}")
        return False

    logger.info(f"\nModel path: {model_path.absolute()}")
    logger.info(f"ONNX files: {[str(f.relative_to(model_path)) for f in onnx_files]}")

    # Load model
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        logger.info("\n✅ Optimum ONNX Runtime imported")

        logger.info("\nLoading ONNX model...")
        model = ORTModelForFeatureExtraction.from_pretrained(
            str(model_path),
            provider="CPUExecutionProvider"
        )
        logger.info("✅ Model loaded successfully")

        logger.info("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        logger.info("✅ Tokenizer loaded successfully")

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Please install: pip install onnxruntime optimum[onnxruntime]")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

    # Test embedding generation
    logger.info("\n" + "-" * 80)
    logger.info("Testing embedding generation")
    logger.info("-" * 80)

    test_texts = [
        "Machine learning is a subset of artificial intelligence",
        "Neural networks are inspired by biological neurons",
        "Deep learning uses multiple layers of neural networks"
    ]

    try:
        logger.info(f"\nGenerating embeddings for {len(test_texts)} texts...")

        # Tokenize
        inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")

        # Generate embeddings
        outputs = model(**inputs)

        # Extract embeddings (mean pooling)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()

        logger.info(f"✅ Embeddings generated!")
        logger.info(f"   Shape: {embeddings.shape}")
        logger.info(f"   Dtype: {embeddings.dtype}")
        logger.info(f"   Dimensions: {embeddings.shape[1]}D")

        # Verify embeddings
        logger.info("\n" + "-" * 80)
        logger.info("Verifying embeddings quality")
        logger.info("-" * 80)

        # Check for NaN/Inf
        has_nan = np.any(np.isnan(embeddings))
        has_inf = np.any(np.isinf(embeddings))

        if has_nan:
            logger.error("❌ Embeddings contain NaN values!")
            return False
        if has_inf:
            logger.error("❌ Embeddings contain Inf values!")
            return False

        logger.info("✅ No NaN/Inf values")

        # Check if all zeros
        is_all_zeros = np.allclose(embeddings, 0.0, atol=1e-6)
        if is_all_zeros:
            logger.error("❌ Embeddings are all zeros!")
            return False

        logger.info("✅ Embeddings are non-zero")

        # Calculate norms
        norms = np.linalg.norm(embeddings, axis=1)
        logger.info(f"   L2 norms: min={norms.min():.4f}, max={norms.max():.4f}, avg={norms.mean():.4f}")

        # Calculate cosine similarity between first two texts
        sim = np.dot(embeddings[0], embeddings[1]) / (norms[0] * norms[1])
        logger.info(f"   Cosine similarity (text1 vs text2): {sim:.4f}")

        if sim < 0.1:
            logger.warning("⚠️  Low similarity between related texts (might indicate quality issue)")

        logger.info("\n" + "=" * 80)
        logger.info("✅ ONNX Model Test Passed!")
        logger.info("=" * 80)
        logger.info("\nModel is ready for use. Next steps:")
        logger.info("  1. Run benchmark: python3 benchmark_onnx.py")
        logger.info("  2. Run MTEB evaluation: python3 evaluate_onnx_mteb.py")
        logger.info("  3. Update main.py to use ONNX model")

        return True

    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution"""
    import sys

    success = test_onnx_model()

    if success:
        logger.info("\n✅ Test successful!")
        sys.exit(0)
    else:
        logger.error("\n❌ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
