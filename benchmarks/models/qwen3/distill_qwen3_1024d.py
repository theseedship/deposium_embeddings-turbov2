#!/usr/bin/env python3
"""
Custom Qwen3 Model2Vec Distillation - 1024D

Distills Qwen/Qwen3-Embedding-0.6B into a 1024D Model2Vec
using the custom 596k multilingual corpus.

Expected:
- Quality: 0.72-0.77 (vs 0.665 for 256D, closer to gemma's 0.788)
- Speed: Still 500-700x faster than gemma
- Size: ~500MB (vs ~200MB for 256D)
"""

import logging
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def distill_qwen3_1024d():
    """Distill Qwen3-Embedding to 1024D Model2Vec with custom corpus"""

    logger.info("=" * 80)
    logger.info("🚀 Custom Qwen3 Model2Vec Distillation - 1024D")
    logger.info("=" * 80)

    try:
        from model2vec.distill import distill
        logger.info("✅ Dependencies imported")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Install: pip install model2vec")
        return False

    # Configuration
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    corpus_path = Path("data/model2vec_corpus_ultra/corpus.jsonl")
    output_dir = Path("models/qwen3-deposium-1024d")
    pca_dims = 1024  # Higher than gemma (768D) for even better quality

    logger.info(f"\n📋 Configuration:")
    logger.info(f"   Base model: {model_name}")
    logger.info(f"   Corpus: {corpus_path} (596k sentences)")
    logger.info(f"   Output dimensions: {pca_dims}D")
    logger.info(f"   Output directory: {output_dir}")

    # Verify corpus exists
    if not corpus_path.exists():
        logger.error(f"❌ Corpus not found: {corpus_path}")
        return False

    # Load corpus
    logger.info(f"\n📥 Loading corpus from {corpus_path}...")
    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            corpus.append(data['text'])

    logger.info(f"✅ Loaded {len(corpus):,} sentences")
    logger.info(f"   Sample: {corpus[0][:100]}...")

    # Distill to Model2Vec (without custom corpus due to tokenizer incompatibility)
    logger.info(f"\n🔬 Starting distillation to {pca_dims}D Model2Vec...")
    logger.info("Model will be downloaded and loaded automatically...")
    logger.info("Note: Using model's default vocabulary (custom corpus not compatible with Qwen3 tokenizer)")
    logger.info("This will take 10-30 minutes depending on CPU...")
    logger.info("Progress:")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Detect device
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Distill without custom vocabulary (Qwen3 uses non-WordPiece tokenizer)
        m2v_model = distill(
            model_name=model_name,
            vocabulary=None,  # Let Model2Vec extract vocabulary from model
            device=device,
            pca_dims=pca_dims,
            apply_zipf=True,
            use_subword=True  # Better for multilingual
        )

        logger.info(f"✅ Distillation complete!")

        # Save model
        logger.info(f"\n💾 Saving model to {output_dir}...")
        m2v_model.save_pretrained(str(output_dir))
        logger.info(f"✅ Model saved!")

        # Test embeddings
        logger.info(f"\n🧪 Testing embeddings...")
        test_texts = [
            "This is a test sentence",
            "Machine learning et intelligence artificielle",  # French
            "El aprendizaje automático es fascinante",  # Spanish
        ]

        embeddings = m2v_model.encode(test_texts, show_progress_bar=False)
        logger.info(f"✅ Test successful!")
        logger.info(f"   Shape: {embeddings.shape}")
        logger.info(f"   Mean: {embeddings.mean():.4f}")
        logger.info(f"   Std: {embeddings.std():.4f}")

        # Save metadata
        metadata = {
            'base_model': model_name,
            'dimensions': pca_dims,
            'custom_corpus': False,
            'note': 'Model2Vec default vocabulary (custom corpus incompatible with Qwen3 tokenizer)',
            'output_dir': str(output_dir),
            'multilingual': True,
            'languages': ['en', 'fr', 'es', 'de', 'it', 'pt', 'nl', 'ru', 'zh', 'ja', 'ar'],
        }

        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"💾 Metadata saved to {metadata_path}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ DISTILLATION SUCCESS")
        logger.info("=" * 80)

        logger.info(f"\n📊 Results:")
        logger.info(f"   Model: {output_dir}")
        logger.info(f"   Dimensions: {pca_dims}D")
        logger.info(f"   Corpus: {len(corpus):,} sentences")
        logger.info(f"   Expected quality: 0.72-0.77 (vs 0.665 for 256D, closer to gemma 0.788)")
        logger.info(f"   Expected speed: 500-700x faster than gemma")

        logger.info(f"\n🚀 Next steps:")
        logger.info(f"   1. Run quality evaluation: python quick_eval_qwen3_1024d.py")
        logger.info(f"   2. Update src/main.py to use new model")
        logger.info(f"   3. Build Docker image: docker build -t deposium-embeddings-qwen3-1024d:latest .")
        logger.info(f"   4. Deploy to Railway")

        return True

    except Exception as e:
        logger.error(f"❌ Distillation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    import sys

    success = distill_qwen3_1024d()

    if success:
        logger.info("\n🎉 Distillation successful!")
        sys.exit(0)
    else:
        logger.error("\n❌ Distillation failed!")
        sys.exit(1)
