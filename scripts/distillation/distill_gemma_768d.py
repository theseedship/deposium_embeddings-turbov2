#!/usr/bin/env python3
"""
Gemma Model2Vec Distillation - Native 768D

Distills google/embeddinggemma-300m into a 768D Model2Vec
using Model2Vec 0.6.0 (which fixes the tokenizer bug).

Expected:
- Quality: 0.70-0.75 (vs 0.665 for Qwen3-256D, closer to gemma's 0.788)
- Speed: 500-700x faster than gemma-int8
- Size: ~400MB
- Native 768D (not forced upscaling like failed 1024D)
"""

import logging
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def distill_gemma_768d():
    """Distill Gemma to native 768D Model2Vec"""

    logger.info("=" * 80)
    logger.info("🚀 Gemma Model2Vec Distillation - Native 768D")
    logger.info("=" * 80)

    try:
        from model2vec.distill import distill
        import model2vec
        logger.info(f"✅ Dependencies imported (Model2Vec {model2vec.__version__})")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Install: pip install model2vec")
        return False

    # Configuration
    model_name = "google/embeddinggemma-300m"
    output_dir = Path("models/gemma-deposium-768d")
    pca_dims = 768  # Native dimension of Gemma

    logger.info(f"\n📋 Configuration:")
    logger.info(f"   Base model: {model_name}")
    logger.info(f"   Output dimensions: {pca_dims}D (native, not forced)")
    logger.info(f"   Output directory: {output_dir}")
    logger.info(f"   Model2Vec version: {model2vec.__version__}")
    logger.info(f"   Expected quality: 0.70-0.75 (baseline: 0.788)")
    logger.info(f"   Expected speed: 500-700x faster than gemma-int8")

    # Distill to Model2Vec
    logger.info(f"\n🔬 Starting distillation to {pca_dims}D Model2Vec...")
    logger.info("Model will be downloaded and loaded automatically...")
    logger.info("This will take 10-30 minutes depending on CPU...")
    logger.info("Progress:")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Detect device
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Distill without custom vocabulary (Gemma uses SentencePiece tokenizer)
        logger.info(f"\n⚙️  Distillation parameters:")
        logger.info(f"   - vocabulary: None (extract from model)")
        logger.info(f"   - device: {device}")
        logger.info(f"   - pca_dims: {pca_dims}")
        logger.info(f"   - apply_zipf: True (deprecated but harmless)")
        logger.info(f"   - use_subword: True (deprecated but harmless)")

        m2v_model = distill(
            model_name=model_name,
            vocabulary=None,  # Let Model2Vec extract vocabulary from model
            device=device,
            pca_dims=pca_dims,
            apply_zipf=True,  # SIF weighting
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
            "Deep learning requires large datasets",  # English (tech)
        ]

        embeddings = m2v_model.encode(test_texts, show_progress_bar=False)
        logger.info(f"✅ Test successful!")
        logger.info(f"   Shape: {embeddings.shape}")
        logger.info(f"   Mean: {embeddings.mean():.4f}")
        logger.info(f"   Std: {embeddings.std():.4f}")

        # Verify dimensions
        if embeddings.shape[1] != pca_dims:
            logger.error(f"❌ Dimension mismatch! Expected {pca_dims}D, got {embeddings.shape[1]}D")
            return False

        # Save metadata
        metadata = {
            'base_model': model_name,
            'dimensions': pca_dims,
            'model2vec_version': model2vec.__version__,
            'custom_corpus': False,
            'note': 'Native 768D Model2Vec from Gemma (0.6.0 fixed tokenizer bug)',
            'output_dir': str(output_dir),
            'baseline_quality': 0.788,
            'expected_quality': '0.70-0.75',
            'expected_speedup': '500-700x vs gemma-int8',
            'multilingual': True,
            'context_length': 2048,
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
        logger.info(f"   Dimensions: {pca_dims}D (native)")
        logger.info(f"   Baseline quality: 0.788 (full Gemma)")
        logger.info(f"   Expected quality: 0.70-0.75")
        logger.info(f"   Expected speed: 500-700x faster than gemma-int8")
        logger.info(f"   Size: ~400MB (vs 1.2GB full Gemma, vs 200MB Qwen3-256D)")

        logger.info(f"\n🚀 Next steps:")
        logger.info(f"   1. Run quality evaluation: python quick_eval_gemma_768d.py")
        logger.info(f"   2. Compare with Qwen3-256D (quality: 0.6651)")
        logger.info(f"   3. Deploy best model to Railway")

        return True

    except Exception as e:
        logger.error(f"❌ Distillation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    import sys

    success = distill_gemma_768d()

    if success:
        logger.info("\n🎉 Distillation successful!")
        sys.exit(0)
    else:
        logger.error("\n❌ Distillation failed!")
        sys.exit(1)
