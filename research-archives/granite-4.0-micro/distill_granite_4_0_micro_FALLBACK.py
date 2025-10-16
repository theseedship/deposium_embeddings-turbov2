#!/usr/bin/env python3
"""
FALLBACK: Distillation granite-4.0-micro STANDARD (si h-micro échoue)
Architecture: Dense transformer classique (40 attention layers)
Use this if h-micro fails due to Mamba2 incompatibility
"""

import torch
from model2vec.distill import distill
from datetime import datetime
from pathlib import Path
import sys

def main():
    print("=" * 80)
    print("🔄 FALLBACK: IBM Granite 4.0 Micro (STANDARD) → Model2Vec 1024D")
    print("=" * 80)
    print()
    print("⚠️  Using STANDARD version (dense transformer)")
    print("   H-Micro failed? This version uses classic architecture.")
    print()

    # GPU check
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available!")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"✅ GPU detected: {gpu_name}")
    print(f"   VRAM: {gpu_mem:.1f}GB")
    print()

    print("📊 Source Model: ibm-granite/granite-4.0-micro (STANDARD)")
    print("   - Size: 3B parameters")
    print("   - Architecture: Dense transformer (40 attention layers)")
    print("   - Embedding dim: 2560 → 1024D")
    print("   - HumanEval: 80%, MMLU: 65.98%, GSM8K: 85.45%")
    print()

    output_dir = Path("granite-4.0-micro-deposium-1024d")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    try:
        print("🔥 Distilling STANDARD Granite 4.0 Micro...")
        print()

        model = distill(
            model_name="ibm-granite/granite-4.0-micro",
            pca_dims=1024,
            device="cuda",
        )

        print()
        print("💾 Saving Model...")
        model.save_pretrained(str(output_dir))

        print("✅ FALLBACK distillation successful!")
        print(f"   Model saved to: {output_dir}/")
        print()

    except Exception as e:
        print(f"❌ FALLBACK also failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
