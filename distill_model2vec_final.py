#!/usr/bin/env python3
"""
FINAL Model2Vec Distillation (Correct API)

Uses the correct model2vec API: distill(model_name, vocabulary, pca_dims)
"""

import logging
from pathlib import Path
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def distill_model2vec_final(output_dir: Path, pca_dims: int = 768):
    """
    Distill gemma to Model2Vec using CORRECT API

    Args:
        output_dir: Output directory for distilled model
        pca_dims: Embedding dimensions (768=no compression)
    """

    logger.info("=" * 80)
    logger.info("🧪 Model2Vec Distillation (FINAL - Correct API)")
    logger.info("=" * 80)

    try:
        from model2vec.distill import distill
        logger.info("✅ model2vec imported")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

    base_model = "google/embeddinggemma-300m"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n📋 Configuration:")
    logger.info(f"  Base model: {base_model}")
    logger.info(f"  Vocabulary: Tokenizer built-in")
    logger.info(f"  PCA dimensions: {pca_dims}D")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Expected output size: ~200MB")
    logger.info(f"  Expected speedup: 100x+")

    try:
        logger.info("\n🔬 Starting distillation with correct API...")
        logger.info("This may take 10-30 minutes...")

        # Use CORRECT API: model_name (string), vocabulary (None), pca_dims
        m2v_model = distill(
            model_name=base_model,
            vocabulary=None,  # Use tokenizer's vocabulary
            pca_dims=pca_dims,
            apply_zipf=True,
            use_subword=True,
        )

        logger.info("✅ Distillation complete!")

        # Save the model
        logger.info(f"\n💾 Saving Model2Vec model to {output_dir}...")
        m2v_model.save_pretrained(str(output_dir))

        # Calculate model size
        model_size = sum(f.stat().st_size for f in output_dir.glob("**/*") if f.is_file())
        model_size_mb = model_size / (1024 * 1024)

        logger.info(f"✅ Model saved ({model_size_mb:.1f} MB)")

        # Test the model
        logger.info("\n🧪 Testing Model2Vec model...")
        test_texts = [
            "This is a test sentence",
            "Machine learning is fascinating",
            "Le droit français est complexe",  # French
            "La física cuántica es interesante",  # Spanish
        ]

        embeddings = m2v_model.encode(test_texts)
        logger.info(f"✅ Generated embeddings: {embeddings.shape}")

        # Sample embeddings to verify
        logger.info(f"\n📊 Sample embedding statistics:")
        logger.info(f"   Mean: {embeddings.mean():.4f}")
        logger.info(f"   Std: {embeddings.std():.4f}")
        logger.info(f"   Min: {embeddings.min():.4f}")
        logger.info(f"   Max: {embeddings.max():.4f}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ DISTILLATION SUCCESSFUL")
        logger.info("=" * 80)

        logger.info(f"\n📊 Summary:")
        logger.info(f"  Model size: {model_size_mb:.1f} MB")
        logger.info(f"  Embedding dim: {embeddings.shape[1]}D")
        logger.info(f"  Expected speedup: 100x+")
        logger.info(f"  Multilingual: Yes (100+ languages)")

        logger.info(f"\n🚀 Next steps:")
        logger.info(f"  1. Quick benchmark: python3 benchmark_model2vec.py")
        logger.info(f"  2. MTEB evaluation: python3 evaluate_model2vec_mteb.py")
        logger.info(f"  3. If MTEB >0.65: Deploy to Railway")

        return True

    except Exception as e:
        logger.error(f"❌ Distillation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main execution"""
    import sys

    parser = argparse.ArgumentParser(description="Distill gemma to Model2Vec (FINAL)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./models/gemma-model2vec-final"),
        help="Output directory"
    )
    parser.add_argument(
        "--pca-dims",
        type=int,
        default=768,
        help="PCA dimensions (768=max quality)"
    )

    args = parser.parse_args()

    success = distill_model2vec_final(
        output_dir=args.output_dir,
        pca_dims=args.pca_dims
    )

    if success:
        logger.info("\n🎉 Model2Vec distillation successful!")
        sys.exit(0)
    else:
        logger.error("\n❌ Model2Vec distillation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
