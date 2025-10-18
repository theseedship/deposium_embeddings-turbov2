#!/usr/bin/env python3
"""
Distill Qwen2.5-7B-Instruct to Model2Vec (1024D)

Target Performance: 91-95% quality
Expected Size: 65MB (32K vocab)
Distillation Time: 2-4 hours (GPU recommended)
"""

import os
import torch
from model2vec import distill_model
from datetime import datetime
from pathlib import Path

print("=" * 80)
print("🚀 Qwen2.5-7B-Instruct → Model2Vec Distillation")
print("=" * 80)
print()

# Configuration
CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "output_name": "qwen25-7b-deposium-1024d",
    "output_dir": "models/qwen25-7b-deposium-1024d",
    "dimensions": 1024,
    "pca_dims": 1024,  # Final dimensions after PCA
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Distillation settings
    "distillation_method": "weighted",  # or "simple" for faster
    "num_layers": -1,  # Use all layers (-1) for best quality
    "apply_pca": True,
    "apply_zipf": True,

    # Training corpus (optional - uses model2vec defaults if None)
    "corpus_path": None,  # or path to custom corpus
    "corpus_size": 1_000_000,  # 1M sentences (default)
}

print("📋 Configuration:")
print("-" * 80)
for key, value in CONFIG.items():
    print(f"  {key:25s}: {value}")
print()

# Check CUDA availability
if CONFIG["device"] == "cuda":
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    print("⚠️  No GPU detected - distillation will be SLOW (10-20+ hours)")
    print("   Recommendation: Use a machine with CUDA GPU")

    # Ask for confirmation
    response = input("\n   Continue without GPU? (yes/no): ")
    if response.lower() != "yes":
        print("Aborting.")
        exit(0)

print()

# Create output directory
output_dir = Path(CONFIG["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

print("🔥 Starting distillation...")
print("-" * 80)
print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

start_time = datetime.now()

try:
    # Distill the model
    model = distill_model(
        model_name=CONFIG["model_name"],
        pca_dims=CONFIG["pca_dims"],
        apply_pca=CONFIG["apply_pca"],
        use_subword=True,  # Use subword embeddings (better for rare words)
        apply_zipf=CONFIG["apply_zipf"],
        device=CONFIG["device"],
        show_progress_bar=True,
    )

    end_time = datetime.now()
    duration = end_time - start_time

    print()
    print("=" * 80)
    print("✅ Distillation complete!")
    print("=" * 80)
    print(f"⏱️  Duration: {duration}")
    print()

    # Save the model
    print("💾 Saving model...")
    model.save_pretrained(str(output_dir))

    # Get model size
    model_size = sum(f.stat().st_size for f in output_dir.glob("**/*") if f.is_file())
    model_size_mb = model_size / (1024 * 1024)

    print(f"✅ Model saved to: {output_dir}")
    print(f"📦 Model size: {model_size_mb:.1f}MB")
    print()

    # Test the model
    print("🧪 Testing model...")
    print("-" * 80)

    test_sentences = [
        "What is machine learning?",
        "Explain neural networks",
        "How to implement gradient descent?",
        "Python programming tutorial"
    ]

    embeddings = model.encode(test_sentences, show_progress_bar=False)

    print(f"✅ Generated embeddings for {len(test_sentences)} test sentences")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dtype: {embeddings.dtype}")
    print(f"   Sample values: {embeddings[0][:5]}")
    print()

    # Calculate embedding norms
    import numpy as np
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"   Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
    print()

    # Write metadata
    metadata_file = output_dir / "distillation_metadata.txt"
    with open(metadata_file, "w") as f:
        f.write(f"Model: {CONFIG['model_name']}\n")
        f.write(f"Output: {CONFIG['output_name']}\n")
        f.write(f"Dimensions: {CONFIG['pca_dims']}\n")
        f.write(f"Device: {CONFIG['device']}\n")
        f.write(f"Distillation time: {duration}\n")
        f.write(f"Model size: {model_size_mb:.1f}MB\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")

    print("=" * 80)
    print("🎉 SUCCESS!")
    print("=" * 80)
    print()
    print("📁 Next steps:")
    print("   1. Test model: python3 test_qwen25_7b_model.py")
    print("   2. Run evaluation: python3 quick_eval_qwen25_7b_1024d.py")
    print("   3. Update API: Update api.py to use new model")
    print("   4. Build Docker: docker build -t deposium-embeddings-v11 .")
    print()

except Exception as e:
    end_time = datetime.now()
    duration = end_time - start_time

    print()
    print("=" * 80)
    print("❌ DISTILLATION FAILED")
    print("=" * 80)
    print(f"Error: {e}")
    print(f"Duration before failure: {duration}")
    print()

    import traceback
    print("Traceback:")
    print(traceback.format_exc())
    print()

    raise
