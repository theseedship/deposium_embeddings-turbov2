#!/usr/bin/env python3
"""
Download Pre-Converted ONNX INT8 Model

Downloads electroglyph/embeddinggemma-300m-ONNX-uint8 from HuggingFace.
This avoids the Gemma3 architecture export issues.
"""

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_onnx_model():
    """Download pre-converted ONNX INT8 model from HuggingFace"""

    try:
        from huggingface_hub import snapshot_download
        logger.info("✅ huggingface_hub imported successfully")
    except ImportError:
        logger.error("❌ huggingface_hub not installed")
        logger.error("Please install: pip install huggingface-hub")
        return False

    logger.info("=" * 80)
    logger.info("📥 Downloading Pre-Converted ONNX INT8 Model")
    logger.info("=" * 80)

    # Use official onnx-community version (FP32, but we can quantize later)
    model_id = "onnx-community/embeddinggemma-300m-ONNX"
    output_dir = Path("./models/gemma-onnx-int8")

    logger.info(f"\nModel: {model_id}")
    logger.info(f"Output: {output_dir.absolute()}")

    try:
        logger.info("\nDownloading official ONNX community version...")
        logger.info("(This is FP32 - we may need to quantize it manually)")

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
        logger.info(f"     python3 test_onnx_model.py")
        logger.info(f"  2. Benchmark performance:")
        logger.info(f"     python3 benchmark_onnx.py")

        return True

    except Exception as e:
        logger.error(f"❌ Download failed: {e}")

        # Try fallback to official ONNX export (FP32)
        logger.info("\n⚠️  Trying fallback: onnx-community/embeddinggemma-300m-ONNX")

        try:
            model_id = "onnx-community/embeddinggemma-300m-ONNX"
            logger.info(f"Downloading {model_id}...")

            downloaded_path = snapshot_download(
                repo_id=model_id,
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )

            logger.info(f"✅ Fallback model downloaded to: {downloaded_path}")
            logger.info("⚠️  Note: This is FP32, not INT8. You may need to quantize it manually.")

            return True

        except Exception as e2:
            logger.error(f"❌ Fallback download also failed: {e2}")
            return False


def main():
    """Main execution"""
    import sys

    success = download_onnx_model()

    if success:
        logger.info("\n🎉 Model download successful!")
        sys.exit(0)
    else:
        logger.error("\n❌ Model download failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
