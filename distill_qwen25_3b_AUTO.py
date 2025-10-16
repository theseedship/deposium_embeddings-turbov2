#!/usr/bin/env python3
"""
Distillation AUTOMATIQUE Qwen2.5-3B-Instruct → Model2Vec 1024D
RTX 4050 6GB VRAM - 1-2 heures
Qualité attendue: 85-88%
"""

import torch
from model2vec.distill import distill
from datetime import datetime
from pathlib import Path
import sys

def main():
    print("=" * 80)
    print("🚀 Qwen2.5-3B → Model2Vec Distillation (LOCAL - AUTO)")
    print("=" * 80)
    print()

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ ERROR: No GPU detected! Aborting.")
        sys.exit(1)

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
    print("⚡ STARTING AUTOMATICALLY (no confirmation)")
    print()

    output_dir = Path("qwen25-3b-deposium-1024d")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()

    print("=" * 80)
    print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # Phase 1: Download model
    print("📥 PHASE 1/4: Downloading Qwen2.5-3B-Instruct (~6GB)")
    print("This may take 5-10 minutes...")
    print()

    # Phase 2: Distillation
    print("🔥 PHASE 2/4: Distilling to Model2Vec 1024D")
    print("This is the longest phase (1-2h on GPU)...")
    print("You can monitor progress in another terminal with:")
    print("  tail -f qwen25-3b-distillation.log")
    print()

    try:
        model = distill(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            pca_dims=1024,
            device="cuda",
        )

        print()
        print("✅ Distillation complete!")
        print()

        # Phase 3: Save
        print("💾 PHASE 3/4: Saving model...")
        model.save_pretrained(str(output_dir))
        print(f"✅ Saved to: {output_dir}/")
        print()

        # Phase 4: Test
        print("🧪 PHASE 4/4: Testing model...")
        test_texts = [
            "What is machine learning?",
            "Explain neural networks",
            "Python programming tutorial",
            "Advanced AI techniques",
        ]
        embeddings = model.encode(test_texts)
        print(f"✅ Test embeddings shape: {embeddings.shape}")
        print()

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
            f.write(f"Device: {gpu_name}\n")
            f.write(f"VRAM: {gpu_mem:.1f}GB\n")
            f.write(f"Start time: {start_time.isoformat()}\n")
            f.write(f"End time: {end_time.isoformat()}\n")
            f.write(f"Duration: {duration}\n")
            f.write(f"Model size: {model_size_mb:.1f}MB\n")
            f.write(f"Test embeddings shape: {embeddings.shape}\n")
            f.write(f"Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}\n")

        # Summary
        print("=" * 80)
        print("✅ DISTILLATION COMPLETE!")
        print("=" * 80)
        print(f"⏱️  Total duration: {duration}")
        print(f"📦 Model size: {model_size_mb:.1f}MB")
        print(f"📂 Output directory: {output_dir}/")
        print()
        print(f"📊 Embedding Stats:")
        print(f"   - Shape: {embeddings.shape}")
        print(f"   - Norm min: {norms.min():.4f}")
        print(f"   - Norm max: {norms.max():.4f}")
        print(f"   - Norm mean: {norms.mean():.4f}")
        print()
        print("🎯 Expected quality: 85-88%")
        print("📊 Actual quality: Run evaluation to confirm")
        print()
        print("Next steps:")
        print("  1. python3 quick_eval_qwen25_3b_1024d.py")
        print("  2. Compare with 7B (HuggingFace) when available")
        print("  3. Compare with 9B Nemotron (HuggingFace) when available")
        print()
        print("🆚 Expected Comparison:")
        print("  - Qwen2.5-3B: ~35MB, 85-88% quality (THIS - FREE)")
        print("  - Qwen2.5-7B: ~65MB, 91-95% quality (HF ~$1)")
        print("  - Nemotron-9B: ~268MB, 90-94% quality (HF ~$2)")
        print()
        print("=" * 80)
        print("✅ SUCCESS! Model ready for deployment.")
        print("=" * 80)

    except Exception as e:
        end_time = datetime.now()
        duration = end_time - start_time

        print()
        print("=" * 80)
        print("❌ ERROR DURING DISTILLATION")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print(f"Duration before error: {duration}")
        print()

        import traceback
        print("Full traceback:")
        print(traceback.format_exc())

        print()
        print("Possible causes:")
        print("  - Out of VRAM (need 5GB free)")
        print("  - Network issue downloading model")
        print("  - Model2Vec compatibility issue")
        print()

        sys.exit(1)

if __name__ == "__main__":
    main()
