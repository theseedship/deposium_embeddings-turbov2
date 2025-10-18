#!/usr/bin/env python3
"""
Distill ColBERT to Model2Vec (static embeddings)

Creates two versions:
1. 384D (native dimension) - smaller, faster
2. 1024D (qwen25-compatible) - drop-in replacement
"""

import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("🔄 Distilling ColBERT to Model2Vec")
    logger.info("=" * 80)

    # Import model2vec
    try:
        from model2vec import StaticModel
    except ImportError:
        logger.error("❌ model2vec not installed!")
        logger.error("   Install with: pip install model2vec")
        return False

    model_name = "mixedbread-ai/mxbai-edge-colbert-v0-32m"

    # ========================================================================
    # Version 1: 384D (Native Dimension)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("📦 Distillation 1: 384D (Native)")
    logger.info("=" * 80)

    output_dir_384 = Path("mxbai-colbert-m2v-384d")

    logger.info(f"\nModel: {model_name}")
    logger.info(f"Target dimensions: 384D")
    logger.info(f"Output directory: {output_dir_384}")

    start_time = time.time()

    try:
        logger.info("\n🔄 Starting distillation (this may take 5-10 minutes)...")

        # Distill with native dimensions
        model_384 = StaticModel.from_sentence_transformers(
            path=model_name,
            dimensionality=384,  # Native dimension
        )

        distill_time = time.time() - start_time

        logger.info(f"✅ Distillation complete in {distill_time:.2f}s")

        # Save model
        logger.info(f"\n💾 Saving to {output_dir_384}...")
        model_384.save_pretrained(str(output_dir_384))

        logger.info("✅ 384D model saved!")

        # Get model size
        import os
        total_size = sum(
            os.path.getsize(output_dir_384 / f)
            for f in output_dir_384.iterdir()
            if f.is_file()
        )
        size_mb = total_size / 1024 / 1024

        logger.info(f"   Model size: {size_mb:.1f} MB")

    except Exception as e:
        logger.error(f"❌ 384D distillation failed: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # Version 2: 1024D (qwen25-compatible)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("📦 Distillation 2: 1024D (qwen25-compatible)")
    logger.info("=" * 80)

    output_dir_1024 = Path("mxbai-colbert-m2v-1024d")

    logger.info(f"\nModel: {model_name}")
    logger.info(f"Target dimensions: 1024D")
    logger.info(f"Output directory: {output_dir_1024}")

    start_time = time.time()

    try:
        logger.info("\n🔄 Starting distillation (this may take 5-10 minutes)...")

        # Distill with 1024D for qwen25 compatibility
        model_1024 = StaticModel.from_sentence_transformers(
            path=model_name,
            dimensionality=1024,  # qwen25 compatible
        )

        distill_time = time.time() - start_time

        logger.info(f"✅ Distillation complete in {distill_time:.2f}s")

        # Save model
        logger.info(f"\n💾 Saving to {output_dir_1024}...")
        model_1024.save_pretrained(str(output_dir_1024))

        logger.info("✅ 1024D model saved!")

        # Get model size
        total_size = sum(
            os.path.getsize(output_dir_1024 / f)
            for f in output_dir_1024.iterdir()
            if f.is_file()
        )
        size_mb = total_size / 1024 / 1024

        logger.info(f"   Model size: {size_mb:.1f} MB")

    except Exception as e:
        logger.error(f"❌ 1024D distillation failed: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("📊 DISTILLATION SUMMARY")
    logger.info("=" * 80)

    if output_dir_384.exists():
        logger.info(f"\n✅ 384D Model: {output_dir_384}")
        logger.info("   Use case: Smaller, faster, edge-optimized")

    if output_dir_1024.exists():
        logger.info(f"\n✅ 1024D Model: {output_dir_1024}")
        logger.info("   Use case: Drop-in replacement for qwen25-1024d")

    logger.info("\n📋 Next steps:")
    logger.info("  1. Test quality of distilled models")
    logger.info("  2. Compare with qwen25-1024d")
    logger.info("  3. Decide which version to use")

    logger.info("\n✅ Distillation complete!")

    return True

if __name__ == "__main__":
    main()
