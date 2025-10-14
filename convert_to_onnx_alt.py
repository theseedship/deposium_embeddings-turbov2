#!/usr/bin/env python3
"""
Alternative ONNX INT8 Conversion (Without Optimum)

Uses standard transformers + onnx export + manual quantization
Avoids Optimum dependency issues.
"""

import logging
import time
from pathlib import Path
import sys
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def export_to_onnx():
    """Export EmbeddingGemma to ONNX format using torch.onnx"""

    logger.info("=" * 80)
    logger.info("🔧 ONNX Export: EmbeddingGemma-300m (Alternative Method)")
    logger.info("=" * 80)

    # Output directory
    output_dir = Path("./models/gemma-onnx-alt")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nOutput directory: {output_dir.absolute()}")

    # Load model and tokenizer
    logger.info("\n" + "-" * 80)
    logger.info("Step 1: Loading model from HuggingFace")
    logger.info("-" * 80)

    try:
        start_time = time.time()

        model = AutoModel.from_pretrained(
            "google/embeddinggemma-300m",
            torch_dtype=torch.float32,  # Use float32 for ONNX export
            trust_remote_code=True
        )
        model.eval()  # Set to eval mode
        model = model.cpu()  # Move to CPU for export

        tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")

        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded in {load_time:.1f}s")

    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

    # Prepare dummy input for export
    logger.info("\n" + "-" * 80)
    logger.info("Step 2: Preparing dummy input for export")
    logger.info("-" * 80)

    try:
        dummy_text = "This is a test sentence for ONNX export."
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512  # Use shorter sequence for export
        )

        # Move inputs to CPU
        inputs = {k: v.cpu() for k, v in inputs.items()}

        logger.info(f"✅ Dummy input prepared:")
        logger.info(f"   input_ids shape: {inputs['input_ids'].shape}")
        logger.info(f"   attention_mask shape: {inputs['attention_mask'].shape}")

    except Exception as e:
        logger.error(f"❌ Input preparation failed: {e}")
        return False

    # Export to ONNX
    logger.info("\n" + "-" * 80)
    logger.info("Step 3: Exporting to ONNX format")
    logger.info("-" * 80)

    onnx_path = output_dir / "model.onnx"

    try:
        start_time = time.time()

        logger.info("Exporting (this may take 2-5 minutes)...")

        # Export using torch.onnx
        torch.onnx.export(
            model,
            (inputs['input_ids'], inputs['attention_mask']),
            str(onnx_path),
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'sequence'},
                'attention_mask': {0: 'batch', 1: 'sequence'},
                'last_hidden_state': {0: 'batch', 1: 'sequence'}
            },
            opset_version=14,  # Use opset 14 for better compatibility
            do_constant_folding=True,
            verbose=False
        )

        export_time = time.time() - start_time
        logger.info(f"✅ ONNX export complete in {export_time:.1f}s")

    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Quantize to INT8
    logger.info("\n" + "-" * 80)
    logger.info("Step 4: Applying INT8 Dynamic Quantization")
    logger.info("-" * 80)

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        start_time = time.time()

        quantized_path = output_dir / "model_int8.onnx"

        logger.info("Quantizing to INT8 (this may take 1-2 minutes)...")

        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QInt8,  # INT8 quantization
            optimize_model=True,  # Apply graph optimizations
            per_channel=True,  # Per-channel quantization (better quality)
            reduce_range=False,  # Full INT8 range
            extra_options={
                'EnableSubgraph': True,
                'ForceQuantizeNoInputCheck': False
            }
        )

        quant_time = time.time() - start_time
        logger.info(f"✅ INT8 quantization complete in {quant_time:.1f}s")

        # Remove unquantized model to save space
        logger.info("Removing unquantized model...")
        onnx_path.unlink()
        logger.info("✅ Cleaned up temporary files")

    except ImportError as e:
        logger.error(f"❌ Quantization library not available: {e}")
        logger.warning("⚠️  Keeping unquantized FP32 model instead")
    except Exception as e:
        logger.error(f"❌ Quantization failed: {e}")
        logger.warning("⚠️  Keeping unquantized FP32 model instead")
        import traceback
        traceback.print_exc()

    # Save tokenizer
    logger.info("\n" + "-" * 80)
    logger.info("Step 5: Saving tokenizer")
    logger.info("-" * 80)

    try:
        tokenizer.save_pretrained(str(output_dir))
        logger.info("✅ Tokenizer saved")
    except Exception as e:
        logger.error(f"⚠️  Tokenizer save failed: {e}")

    # Verify output files
    logger.info("\n" + "-" * 80)
    logger.info("Step 6: Verifying output files")
    logger.info("-" * 80)

    model_files = list(output_dir.glob("*.onnx"))
    config_files = list(output_dir.glob("*.json"))

    if model_files:
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            logger.info(f"  ✅ {model_file.name}: {size_mb:.1f} MB")
    else:
        logger.error("  ❌ No ONNX model files found!")
        return False

    if config_files:
        logger.info(f"  ✅ {len(config_files)} config files found")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_dir.glob("*") if f.is_file())
    total_size_mb = total_size / (1024 * 1024)

    logger.info(f"\n📊 Total model size: {total_size_mb:.1f} MB")

    # Success summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ ONNX Export Complete!")
    logger.info("=" * 80)
    logger.info(f"\nModel saved to: {output_dir.absolute()}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Test model loading:")
    logger.info(f"     python3 test_onnx_model_alt.py")
    logger.info(f"  2. Benchmark performance:")
    logger.info(f"     python3 benchmark_onnx_alt.py")

    return True


def main():
    """Main execution"""
    success = export_to_onnx()

    if success:
        logger.info("\n🎉 Conversion successful!")
        sys.exit(0)
    else:
        logger.error("\n❌ Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
