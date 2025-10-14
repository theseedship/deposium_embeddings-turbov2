#!/usr/bin/env python3
"""
Fixed Model2Vec Distillation from Gemma

Uses correct model2vec API with vocabulary extraction from corpus.
"""

import logging
import json
from pathlib import Path
import argparse
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_vocabulary_from_corpus(corpus_path: Path, max_vocab: int = 50000):
    """Extract most common vocabulary from corpus"""
    logger.info(f"📂 Extracting vocabulary from {corpus_path}")

    # Load corpus
    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            corpus.append(data['text'])

    logger.info(f"✅ Loaded {len(corpus)} sentences")

    # Tokenize and count words (simple whitespace tokenization)
    word_counts = Counter()
    for text in corpus:
        words = text.lower().split()
        word_counts.update(words)

    # Get top vocabulary
    vocab = [word for word, count in word_counts.most_common(max_vocab)]

    logger.info(f"✅ Extracted vocabulary: {len(vocab)} words")
    logger.info(f"   Most common: {', '.join(vocab[:20])}")

    return vocab


def distill_model2vec_fixed(corpus_path: Path, output_dir: Path, pca_dims: int = 768):
    """
    Distill gemma to Model2Vec using CORRECT API

    Args:
        corpus_path: Path to corpus JSONL file
        output_dir: Output directory for distilled model
        pca_dims: Embedding dimensions (768=no compression, 256=default)
    """

    logger.info("=" * 80)
    logger.info("🧪 Model2Vec Distillation (FIXED API)")
    logger.info("=" * 80)

    # Import required libraries
    try:
        from model2vec.distill import distill_from_model
        from transformers import AutoModel, AutoTokenizer
        logger.info("✅ model2vec and transformers imported")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

    # Extract vocabulary from corpus
    vocabulary = extract_vocabulary_from_corpus(corpus_path, max_vocab=50000)

    # Configuration
    base_model = "google/embeddinggemma-300m"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n📋 Configuration:")
    logger.info(f"  Base model: {base_model}")
    logger.info(f"  Vocabulary size: {len(vocabulary)} words")
    logger.info(f"  PCA dimensions: {pca_dims}D")
    logger.info(f"  Output dir: {output_dir}")

    try:
        logger.info("\n⏳ Loading base model and tokenizer...")
        model = AutoModel.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        logger.info("✅ Model and tokenizer loaded")

        logger.info("\n🔬 Starting distillation with vocabulary...")
        logger.info("This may take 10-30 minutes...")

        # Distill using vocabulary from corpus
        m2v_model = distill_from_model(
            model=model,
            tokenizer=tokenizer,
            vocabulary=vocabulary,  # Pass our extracted vocabulary
            pca_dims=pca_dims,
            apply_zipf=True,  # Apply Zipf's law weighting
            use_subword=True,  # Use subword information
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

        logger.info("\n" + "=" * 80)
        logger.info("✅ DISTILLATION SUCCESSFUL")
        logger.info("=" * 80)

        logger.info(f"\n📊 Summary:")
        logger.info(f"  Model size: {model_size_mb:.1f} MB")
        logger.info(f"  Embedding dim: {embeddings.shape[1]}D")
        logger.info(f"  Vocabulary: {len(vocabulary)} words")

        logger.info(f"\n🚀 Next steps:")
        logger.info(f"  1. Benchmark: python3 benchmark_model2vec.py")
        logger.info(f"  2. MTEB eval: python3 evaluate_model2vec_mteb.py")

        return True

    except Exception as e:
        logger.error(f"❌ Distillation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main execution"""
    import sys

    parser = argparse.ArgumentParser(description="Distill gemma to Model2Vec (FIXED)")
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("./data/model2vec_corpus_ultra/corpus.jsonl"),
        help="Path to corpus JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./models/gemma-model2vec-fixed"),
        help="Output directory for distilled model"
    )
    parser.add_argument(
        "--pca-dims",
        type=int,
        default=768,
        help="PCA dimensions (768=max quality, 256=default)"
    )

    args = parser.parse_args()

    if not args.corpus_path.exists():
        logger.error(f"❌ Corpus not found: {args.corpus_path}")
        sys.exit(1)

    success = distill_model2vec_fixed(
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
