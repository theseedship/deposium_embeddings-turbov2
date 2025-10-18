#!/usr/bin/env python3
"""
Optimized Model2Vec Distillation from Gemma

Uses corpus-based distillation (not vocabulary-only) for higher quality.

Key optimizations:
1. Corpus-based approach: Learn from real usage patterns
2. Diverse multilingual + legal corpus
3. Optimal parameters for quality-speed balance
4. 768D embeddings (no compression for max quality)

Expected results:
- Speed: 100x+ faster than PyTorch INT8
- Quality: 0.65-0.75 MTEB (vs 0.788 baseline)
- Size: ~200MB (vs 300MB PyTorch INT8)
"""

import logging
import json
from pathlib import Path
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_corpus(corpus_path: Path, max_sentences: int = 500000):
    """Load corpus from JSONL file"""
    logger.info(f"📂 Loading corpus from {corpus_path}")

    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_sentences and i >= max_sentences:
                break
            data = json.loads(line)
            corpus.append(data['text'])

    logger.info(f"✅ Loaded {len(corpus)} sentences")
    logger.info(f"   Avg length: {sum(len(s) for s in corpus) / len(corpus):.1f} chars")

    return corpus


def distill_model2vec_optimized(corpus_path: Path, output_dir: Path, pca_dims: int = 768):
    """
    Distill gemma to Model2Vec using corpus-based approach

    Args:
        corpus_path: Path to corpus JSONL file
        output_dir: Output directory for distilled model
        pca_dims: Embedding dimensions (768=no compression, 512=faster)
    """

    logger.info("=" * 80)
    logger.info("🧪 OPTIMIZED Model2Vec Distillation")
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

    # Load corpus
    corpus = load_corpus(corpus_path)

    # Configuration
    base_model = "google/embeddinggemma-300m"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n📋 Configuration:")
    logger.info(f"  Base model: {base_model}")
    logger.info(f"  Corpus size: {len(corpus)} sentences")
    logger.info(f"  PCA dimensions: {pca_dims}D")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Approach: Corpus-based (NOT vocabulary-only)")
    logger.info(f"  Expected output size: ~200MB")
    logger.info(f"  Expected speedup: 100x+")

    try:
        logger.info("\n⏳ Loading base model (this may take a minute)...")
        model = SentenceTransformer(base_model)
        logger.info("✅ Base model loaded")

        logger.info("\n🔬 Starting corpus-based distillation...")
        logger.info("This will:")
        logger.info("  1. Analyze corpus for token usage patterns")
        logger.info("  2. Extract vocabulary from corpus")
        logger.info("  3. Compute embeddings for tokens in context")
        logger.info("  4. Create static lookup table")
        logger.info("  5. Apply PCA dimensionality reduction (optional)")
        logger.info("\nThis may take 30-60 minutes depending on corpus size...")

        # Distill using corpus-based approach
        # This is MUCH better than vocabulary-only because it captures
        # how words are actually used together in real text
        m2v_model = distill(
            model=model,
            vocabulary=None,  # Extract vocabulary from corpus
            corpus=corpus,    # Use corpus for context-aware embeddings
            pca_dims=pca_dims,  # Keep 768D for max quality (or 512D for speed)
            apply_pca=True,   # Apply PCA for optimization
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
            "Le droit français est complexe",  # French legal
            "La física cuántica es interesante",  # Spanish science
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
        logger.info(f"  Corpus sentences: {len(corpus)}")
        logger.info(f"  Expected speedup: 100x+")
        logger.info(f"  Multilingual: Yes (100+ languages)")
        logger.info(f"  Domains: Scientific, Legal, General")

        logger.info(f"\n🚀 Next steps:")
        logger.info(f"  1. Quick benchmark: python3 benchmark_model2vec.py")
        logger.info(f"  2. MTEB evaluation: python3 evaluate_model2vec_mteb.py")
        logger.info(f"  3. If MTEB >0.65: Deploy to Railway")
        logger.info(f"  4. If MTEB <0.65: Consider larger corpus or different approach")

        return True

    except Exception as e:
        logger.error(f"❌ Distillation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main execution"""
    import sys

    parser = argparse.ArgumentParser(description="Distill gemma to Model2Vec")
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("./data/model2vec_corpus_ultra/corpus.jsonl"),
        help="Path to corpus JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./models/gemma-model2vec-optimized"),
        help="Output directory for distilled model"
    )
    parser.add_argument(
        "--pca-dims",
        type=int,
        default=768,
        help="PCA dimensions (768=max quality, 512=faster)"
    )

    args = parser.parse_args()

    # Check if corpus exists
    if not args.corpus_path.exists():
        logger.error(f"❌ Corpus not found: {args.corpus_path}")
        logger.error("Run prepare_corpus_ultra.py first!")
        sys.exit(1)

    success = distill_model2vec_optimized(
        corpus_path=args.corpus_path,
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
