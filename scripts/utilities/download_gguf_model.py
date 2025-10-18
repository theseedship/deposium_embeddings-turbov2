#!/usr/bin/env python3
"""
Download GGUF Q4_K_M Model

Downloads sabafallah/embeddinggemma-300m-Q4_K_M-GGUF from HuggingFace.
This is a more aggressively quantized model (Q4 vs INT8) that should be faster.
"""

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_gguf_model():
    """Download GGUF Q4_K_M model from HuggingFace"""

    try:
        from huggingface_hub import snapshot_download
        logger.info("✅ huggingface_hub imported successfully")
    except ImportError:
        logger.error("❌ huggingface_hub not installed")
        logger.error("Please install: pip install huggingface-hub")
        return False

    logger.info("=" * 80)
    logger.info("📥 Downloading GGUF Q4_K_M Model")
    logger.info("=" * 80)

    model_id = "sabafallah/embeddinggemma-300m-Q4_K_M-GGUF"
    output_dir = Path("./models/gemma-gguf-q4")

    logger.info(f"\nModel: {model_id}")
    logger.info(f"Output: {output_dir.absolute()}")
    logger.info("\nThis is a Q4_K_M quantized model (4-bit weights)")
    logger.info("Expected benefits:")
    logger.info("  - Smaller model size (~150MB vs 295MB INT8)")
    logger.info("  - Potentially 2-3x faster than ONNX INT8")
    logger.info("  - llama.cpp backend is highly optimized for CPU")
    logger.info("\nPotential drawbacks:")
    logger.info("  - More aggressive quantization (may lose some quality)")
    logger.info("  - Need to verify MTEB score")

    try:
        logger.info("\nDownloading GGUF model...")

        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )

        logger.info(f"✅ Model downloaded to: {downloaded_path}")

        # List downloaded files
        logger.info("\n📁 Downloaded files:")
        model_files = list(output_dir.glob("**/*"))
        for f in sorted(model_files):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                logger.info(f"  - {f.name}: {size_mb:.1f} MB")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in output_dir.glob("**/*") if f.is_file())
        total_size_mb = total_size / (1024 * 1024)

        logger.info(f"\n📊 Total model size: {total_size_mb:.1f} MB")

        logger.info("\n" + "=" * 80)
        logger.info("✅ Download Complete!")
        logger.info("=" * 80)
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Test model loading:")
        logger.info(f"     python3 test_gguf_model.py")
        logger.info(f"  2. Benchmark performance:")
        logger.info(f"     python3 benchmark_gguf.py")

        return True

    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False


def main():
    """Main execution"""
    import sys

    success = download_gguf_model()

    if success:
        logger.info("\n🎉 Model download successful!")
        sys.exit(0)
    else:
        logger.error("\n❌ Model download failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
