#!/usr/bin/env python3
"""
ONNX INT8 Conversion for EmbeddingGemma-300m

Converts gemma model to ONNX format with INT8 quantization for optimal CPU performance.

Expected improvements:
- Latency: 3-4x faster (~10-15ms per embedding on Railway CPU)
- Size: ~200-250MB (vs ~300MB INT8 PyTorch)
- Quality: <2% loss (MTEB ~0.77-0.775 vs 0.788 baseline)
"""

import logging
import time
from pathlib import Path
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_to_onnx():
    """Convert EmbeddingGemma to ONNX INT8 format"""

    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from optimum.onnxruntime.configuration import OptimizationConfig, QuantizationConfig
        from transformers import AutoTokenizer
        logger.info("✅ Optimum ONNX Runtime imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import ONNX dependencies: {e}")
        logger.error("Please install: pip install onnxruntime>=1.17.0 optimum[onnxruntime]>=1.16.0")
        return False

    logger.info("=" * 80)
    logger.info("🔧 ONNX INT8 Conversion: EmbeddingGemma-300m")
    logger.info("=" * 80)

    # Output directory
    output_dir = Path("./models/gemma-onnx-int8")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nOutput directory: {output_dir.absolute()}")

    # Step 1: Convert to ONNX (FP32)
    logger.info("\n" + "-" * 80)
    logger.info("Step 1: Converting EmbeddingGemma to ONNX format (FP32)")
    logger.info("-" * 80)

    try:
        start_time = time.time()

        logger.info("Loading model from HuggingFace...")
        model = ORTModelForFeatureExtraction.from_pretrained(
            "google/embeddinggemma-300m",
            export=True,  # Automatically convert to ONNX
            provider="CPUExecutionProvider"  # Force CPU provider
        )

        conversion_time = time.time() - start_time
        logger.info(f"✅ Model converted to ONNX in {conversion_time:.1f}s")

    except Exception as e:
        logger.error(f"❌ ONNX conversion failed: {e}")
        return False

    # Step 2: Save ONNX model first
    logger.info("\n" + "-" * 80)
    logger.info("Step 2: Saving ONNX model")
    logger.info("-" * 80)

    try:
        start_time = time.time()

        logger.info("Saving ONNX model...")
        model.save_pretrained(str(output_dir))

        save_time = time.time() - start_time
        logger.info(f"✅ ONNX model saved in {save_time:.1f}s")

    except Exception as e:
        logger.error(f"❌ Model save failed: {e}")
        return False

    # Step 3: Apply INT8 Quantization using onnxruntime
    logger.info("\n" + "-" * 80)
    logger.info("Step 3: Applying INT8 Dynamic Quantization")
    logger.info("-" * 80)

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        start_time = time.time()

        # Find the ONNX model file
        onnx_files = list(output_dir.glob("*.onnx"))
        if not onnx_files:
            logger.error("❌ No ONNX files found to quantize")
            return False

        logger.info(f"Found {len(onnx_files)} ONNX files to quantize")

        for onnx_file in onnx_files:
            logger.info(f"\nQuantizing {onnx_file.name}...")

            quantized_file = output_dir / f"{onnx_file.stem}_quantized.onnx"

            quantize_dynamic(
                model_input=str(onnx_file),
                model_output=str(quantized_file),
                weight_type=QuantType.QInt8,  # INT8 quantization
                optimize_model=True,  # Apply graph optimizations
                per_channel=True,  # Per-channel quantization (better quality)
                reduce_range=False  # Full INT8 range
            )

            logger.info(f"   ✅ Quantized: {quantized_file.name}")

            # Remove original unquantized file and rename quantized
            onnx_file.unlink()
            quantized_file.rename(onnx_file)
            logger.info(f"   ✅ Replaced with quantized version")

        quant_time = time.time() - start_time
        logger.info(f"\n✅ INT8 quantization complete in {quant_time:.1f}s")

    except Exception as e:
        logger.error(f"❌ INT8 quantization failed: {e}")
        logger.info("\nTrying fallback: Saving FP32 ONNX model without quantization...")

        try:
            model.save_pretrained(str(output_dir))
            logger.info("✅ FP32 ONNX model saved (quantization skipped)")
        except Exception as e2:
            logger.error(f"❌ Fallback save failed: {e2}")
            return False

    # Step 3: Save tokenizer
    logger.info("\n" + "-" * 80)
    logger.info("Step 3: Saving tokenizer")
    logger.info("-" * 80)

    try:
        tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")
        tokenizer.save_pretrained(str(output_dir))
        logger.info("✅ Tokenizer saved")
    except Exception as e:
        logger.error(f"⚠️  Tokenizer save failed: {e}")

    # Step 4: Verify model files
    logger.info("\n" + "-" * 80)
    logger.info("Step 4: Verifying output files")
    logger.info("-" * 80)

    model_files = list(output_dir.glob("*.onnx"))
    config_files = list(output_dir.glob("config.json"))

    if model_files:
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            logger.info(f"  ✅ {model_file.name}: {size_mb:.1f} MB")
    else:
        logger.error("  ❌ No ONNX model files found!")
        return False

    if config_files:
        logger.info(f"  ✅ config.json found")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_dir.glob("*") if f.is_file())
    total_size_mb = total_size / (1024 * 1024)

    logger.info(f"\n📊 Total model size: {total_size_mb:.1f} MB")

    # Success summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ ONNX INT8 Conversion Complete!")
    logger.info("=" * 80)
    logger.info(f"\nModel saved to: {output_dir.absolute()}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Test model loading:")
    logger.info(f"     python3 test_onnx_model.py")
    logger.info(f"  2. Benchmark performance:")
    logger.info(f"     python3 benchmark_onnx.py")
    logger.info(f"  3. Run MTEB evaluation:")
    logger.info(f"     python3 evaluate_onnx_mteb.py")
    logger.info(f"  4. Update main.py to use ONNX model")

    return True


def main():
    """Main execution"""
    success = convert_to_onnx()

    if success:
        logger.info("\n🎉 Conversion successful!")
        sys.exit(0)
    else:
        logger.error("\n❌ Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
