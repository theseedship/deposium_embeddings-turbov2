#!/usr/bin/env python3
"""
Distill Gemma to Model2Vec

This script converts the gemma transformer model into a static embedding model
using the Model2Vec distillation technique.

Model2Vec creates a vocabulary-based embedding lookup table by:
1. Extracting the tokenizer vocabulary from gemma
2. Computing embeddings for each token using gemma
3. Storing these as static vectors in a lookup table

Result: 100x faster inference (dictionary lookup vs transformer inference)
"""

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def distill_to_model2vec():
    """Distill gemma to Model2Vec format"""

    logger.info("=" * 80)
    logger.info("🧪 Model2Vec Distillation from Gemma")
    logger.info("=" * 80)

    # Import required libraries
    try:
        from model2vec.distill import distill
        from sentence_transformers import SentenceTransformer
        logger.info("✅ model2vec and sentence_transformers imported")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Install: pip install model2vec sentence-transformers")
        return False

    # Configuration
    base_model = "google/embeddinggemma-300m"
    output_dir = Path("./models/gemma-model2vec")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n📋 Configuration:")
    logger.info(f"  Base model: {base_model}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Expected output size: ~30MB")
    logger.info(f"  Expected speedup: 100x+")

    try:
        logger.info("\n⏳ Loading base model (this may take a minute)...")
        model = SentenceTransformer(base_model)
        logger.info("✅ Base model loaded")

        logger.info("\n🔬 Starting distillation...")
        logger.info("This will:")
        logger.info("  1. Extract vocabulary from tokenizer")
        logger.info("  2. Compute embeddings for each token")
        logger.info("  3. Create static lookup table")
        logger.info("  4. Save as Model2Vec model")
        logger.info("\nThis may take 10-30 minutes depending on vocabulary size...")

        # Distill the model
        # Model2Vec will use the tokenizer vocabulary automatically
        m2v_model = distill(
            model=model,
            vocabulary=None,  # Use tokenizer vocabulary automatically
            pca_dims=768,  # Keep original dimensionality
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
            "Machine learning is fascinating"
        ]

        embeddings = m2v_model.encode(test_texts)
        logger.info(f"✅ Generated embeddings: {embeddings.shape}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ DISTILLATION SUCCESSFUL")
        logger.info("=" * 80)

        logger.info(f"\n📊 Summary:")
        logger.info(f"  Model size: {model_size_mb:.1f} MB")
        logger.info(f"  Embedding dim: {embeddings.shape[1]}D")
        logger.info(f"  Expected speedup: 100x+")

        logger.info(f"\n🚀 Next steps:")
        logger.info(f"  1. Benchmark performance: python3 benchmark_model2vec.py")
        logger.info(f"  2. Run MTEB evaluation: python3 evaluate_model2vec_mteb.py")
        logger.info(f"  3. If quality good (>0.65 MTEB): Deploy to Railway")

        return True

    except Exception as e:
        logger.error(f"❌ Distillation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main execution"""
    import sys

    success = distill_to_model2vec()

    if success:
        logger.info("\n🎉 Model2Vec distillation successful!")
        sys.exit(0)
    else:
        logger.error("\n❌ Model2Vec distillation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
