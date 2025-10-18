#!/usr/bin/env python3
"""
Simple Model2Vec Distillation (No Custom Vocabulary)

Uses tokenizer's built-in vocabulary since gemma uses SentencePiece.
"""

import logging
from pathlib import Path
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def distill_model2vec_simple(output_dir: Path, pca_dims: int = 768):
    """
    Distill gemma to Model2Vec using tokenizer vocabulary

    Args:
        output_dir: Output directory for distilled model
        pca_dims: Embedding dimensions (768=no compression)
    """

    logger.info("=" * 80)
    logger.info("🧪 Model2Vec Distillation (Simple - Tokenizer Vocab)")
    logger.info("=" * 80)

    try:
        from model2vec.distill import distill_from_model
        from transformers import AutoModel, AutoTokenizer
        logger.info("✅ model2vec and transformers imported")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

    base_model = "google/embeddinggemma-300m"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n📋 Configuration:")
    logger.info(f"  Base model: {base_model}")
    logger.info(f"  Vocabulary: Tokenizer built-in (SentencePiece)")
    logger.info(f"  PCA dimensions: {pca_dims}D")
    logger.info(f"  Output dir: {output_dir}")

    try:
        logger.info("\n⏳ Loading base model and tokenizer...")
        model = AutoModel.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        logger.info("✅ Model and tokenizer loaded")
        logger.info(f"   Tokenizer type: {type(tokenizer).__name__}")
        logger.info(f"   Vocab size: {len(tokenizer)}")

        logger.info("\n🔬 Starting distillation...")
        logger.info("This may take 10-30 minutes...")

        # Distill WITHOUT custom vocabulary (use tokenizer vocab)
        m2v_model = distill_from_model(
            model=model,
            tokenizer=tokenizer,
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
            "Le droit français est complexe",
            "La física cuántica es interesante",
        ]

        embeddings = m2v_model.encode(test_texts)
        logger.info(f"✅ Generated embeddings: {embeddings.shape}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ DISTILLATION SUCCESSFUL")
        logger.info("=" * 80)

        logger.info(f"\n📊 Summary:")
        logger.info(f"  Model size: {model_size_mb:.1f} MB")
        logger.info(f"  Embedding dim: {embeddings.shape[1]}D")
        logger.info(f"  Vocabulary: {len(tokenizer)} tokens")

        logger.info(f"\n🚀 Next steps:")
        logger.info(f"  1. Benchmark speed")
        logger.info(f"  2. MTEB evaluation")
        logger.info(f"  3. Deploy if quality >0.65")

        return True

    except Exception as e:
        logger.error(f"❌ Distillation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main execution"""
    import sys

    parser = argparse.ArgumentParser(description="Distill gemma to Model2Vec (Simple)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./models/gemma-model2vec"),
        help="Output directory"
    )
    parser.add_argument(
        "--pca-dims",
        type=int,
        default=768,
        help="PCA dimensions"
    )

    args = parser.parse_args()

    success = distill_model2vec_simple(
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
