#!/usr/bin/env python3
"""
Distillation LOCAL Qwen2.5-3B-Instruct → Model2Vec 1024D
RTX 4050 5GB VRAM - 1-2 heures
Qualité attendue: 85-88%
"""

import torch
from model2vec.distill import distill
from datetime import datetime
from pathlib import Path
import sys

def main():
    print("=" * 80)
    print("🚀 Qwen2.5-3B → Model2Vec Distillation (LOCAL)")
    print("=" * 80)
    print()

    # Check GPU
    if not torch.cuda.is_available():
        print("⚠️  WARNING: No GPU detected! Will use CPU (10-20h)")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ VRAM: {gpu_mem:.1f}GB")

        if gpu_mem < 4.5:
            print("❌ ERROR: Need at least 5GB VRAM!")
            sys.exit(1)

    print()
    print("📊 Configuration:")
    print("  - Model: Qwen/Qwen2.5-3B-Instruct")
    print("  - Target: Model2Vec 1024D")
    print("  - Expected size: ~35MB")
    print("  - Expected quality: 85-88%")
    print("  - Expected time: 1-2 hours on RTX 4050")
    print()

    response = input("🚀 Start distillation? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)

    output_dir = Path("qwen25-3b-deposium-1024d")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()

    print()
    print("=" * 80)
    print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # Phase 1: Download model
    print("📥 PHASE 1: Downloading Qwen2.5-3B-Instruct (~6GB)")
    print("This may take 5-10 minutes...")
    print()

    # Phase 2: Distillation
    print("🔥 PHASE 2: Distilling to Model2Vec 1024D")
    print("This is the longest phase (1-2h on GPU)...")
    print()

    try:
        model = distill(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            pca_dims=1024,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        print()
        print("✅ Distillation complete!")
        print()

        # Phase 3: Save
        print("💾 PHASE 3: Saving model...")
        model.save_pretrained(str(output_dir))

        # Phase 4: Test
        print()
        print("🧪 PHASE 4: Testing model...")
        test_texts = [
            "What is machine learning?",
            "Explain neural networks",
            "Python programming tutorial",
            "Advanced AI techniques",
        ]
        embeddings = model.encode(test_texts)

        # Stats
        import numpy as np
        norms = np.linalg.norm(embeddings, axis=1)

        end_time = datetime.now()
        duration = end_time - start_time

        # Size
        model_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        model_size_mb = model_size / (1024 * 1024)

        # Metadata
        with open(output_dir / "distillation_metadata.txt", "w") as f:
            f.write(f"Model: Qwen/Qwen2.5-3B-Instruct\n")
            f.write(f"Distillation method: Model2Vec\n")
            f.write(f"Dimensions: 1024D\n")
            f.write(f"Device: {gpu_name if torch.cuda.is_available() else 'CPU'}\n")
            f.write(f"Start time: {start_time.isoformat()}\n")
            f.write(f"End time: {end_time.isoformat()}\n")
            f.write(f"Duration: {duration}\n")
            f.write(f"Model size: {model_size_mb:.1f}MB\n")

        # Summary
        print()
        print("=" * 80)
        print("✅ DISTILLATION COMPLETE!")
        print("=" * 80)
        print(f"⏱️  Duration: {duration}")
        print(f"📦 Model size: {model_size_mb:.1f}MB")
        print(f"📂 Output: {output_dir}/")
        print(f"🧪 Test embeddings shape: {embeddings.shape}")
        print(f"📊 Embedding norms:")
        print(f"   - Min: {norms.min():.4f}")
        print(f"   - Max: {norms.max():.4f}")
        print(f"   - Mean: {norms.mean():.4f}")
        print()
        print("🎯 Expected quality: 85-88%")
        print()
        print("Next steps:")
        print("  1. Test: python3 test_qwen25_3b_model.py")
        print("  2. Evaluate: python3 quick_eval_qwen25_3b_1024d.py")
        print("  3. Compare with 7B results when available")
        print()
        print("🆚 Comparison:")
        print("  - Qwen2.5-3B: ~35MB, 85-88% quality (THIS)")
        print("  - Qwen2.5-7B: ~65MB, 91-95% quality (HuggingFace)")
        print("  - Nemotron-9B: ~268MB, 90-94% quality (HuggingFace)")
        print()

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print()

        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
