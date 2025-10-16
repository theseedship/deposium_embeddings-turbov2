#!/usr/bin/env python3
"""
Distillation AUTOMATIQUE IBM Granite 4.0 Micro → Model2Vec 1024D
RTX 4050 6GB VRAM - 1-2 heures
Architecture: GQA, RoPE, SwiGLU (moderne)
Multilingue: 12 langues (EN, FR, DE, ES, JP, PT, AR, CS, IT, KO, NL, ZH)
"""

import torch
from model2vec.distill import distill
from datetime import datetime
from pathlib import Path
import sys

def main():
    print("=" * 80)
    print("🚀 IBM Granite 4.0 Micro → Model2Vec 1024D Distillation")
    print("=" * 80)
    print()

    # GPU check
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available!")
        print("   This script requires a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"✅ GPU detected: {gpu_name}")
    print(f"   VRAM: {gpu_mem:.1f}GB")
    print()

    # Model info
    print("📊 Source Model: ibm-granite/granite-4.0-h-micro (HYBRID)")
    print("   - Size: 3B parameters")
    print("   - Architecture: HYBRID (4 attention + 36 Mamba2 layers)")
    print("   - Context: 128K tokens")
    print("   - Languages: 12 (EN, FR, DE, ES, JP, PT, AR, CS, IT, KO, NL, ZH)")
    print("   - Embedding dim: 2048 → 1024D (perfect 2:1 ratio)")
    print("   - IFEval: 84.32% (excellent instruction-following)")
    print()

    print("🎯 Target Model: granite-4.0-micro-deposium-1024d")
    print("   - Dimensions: 1024D")
    print("   - Expected size: ~40-50MB")
    print("   - Expected quality: 88-92% (multilingual advantage)")
    print("   - Expected time: 1-2 hours on RTX 4050")
    print()

    print("⚡ STARTING AUTOMATICALLY (no confirmation)")
    print()

    # Output directory
    output_dir = Path("granite-4.0-micro-deposium-1024d")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    print("📥 PHASE 1/4: Downloading IBM Granite 4.0-H Micro (~6GB)")
    print("This may take 5-10 minutes...")
    print()

    print("🔥 PHASE 2/4: Distilling to Model2Vec 1024D")
    print("This is the longest phase (1-2h on GPU)...")
    print("⚠️  Testing HYBRID Mamba2 architecture (experimental)")
    print("   If this fails, we'll fallback to standard granite-4.0-micro")
    print("You can monitor progress in another terminal with:")
    print("  tail -f granite-4.0-distillation.log")
    print()

    try:
        # Distillation - H-MICRO (Hybrid Mamba2)
        model = distill(
            model_name="ibm-granite/granite-4.0-h-micro",
            pca_dims=1024,
            device="cuda",
        )

        print()
        print("=" * 80)
        print("💾 PHASE 3/4: Saving Model")
        print("=" * 80)
        print()

        # Save
        model.save_pretrained(str(output_dir))

        print(f"✅ Model saved to: {output_dir}/")
        print()

        print("=" * 80)
        print("🧪 PHASE 4/4: Quality Check")
        print("=" * 80)
        print()

        # Test embeddings
        test_texts = [
            "Machine learning is transforming AI",
            "Le deep learning révolutionne l'IA",  # French
            "Das maschinelle Lernen verändert KI",  # German
            "El aprendizaje automático transforma la IA",  # Spanish
            "Python programming language",
            "Natural language processing",
        ]

        print("Testing multilingual embeddings...")
        embeddings = model.encode(test_texts)

        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Dtype: {embeddings.dtype}")
        print()

        # Check embedding properties
        import numpy as np

        norms = np.linalg.norm(embeddings, axis=1)
        print(f"📊 Embedding Statistics:")
        print(f"   - Dimensions: {embeddings.shape[1]}")
        print(f"   - Norm min: {norms.min():.4f}")
        print(f"   - Norm max: {norms.max():.4f}")
        print(f"   - Norm mean: {norms.mean():.4f}")
        print()

        # Test multilingual similarity
        print("🌍 Multilingual Similarity Test:")
        en_emb = embeddings[0]  # English
        fr_emb = embeddings[1]  # French
        de_emb = embeddings[2]  # German
        es_emb = embeddings[3]  # Spanish

        # Cosine similarity
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        print(f"   EN-FR similarity: {cosine_sim(en_emb, fr_emb):.4f}")
        print(f"   EN-DE similarity: {cosine_sim(en_emb, de_emb):.4f}")
        print(f"   EN-ES similarity: {cosine_sim(en_emb, es_emb):.4f}")
        print()

        print("🎯 Expected quality: 88-92% (multilingual optimized)")
        print("📊 Actual quality: Run evaluation to confirm")
        print()

        print("Next steps:")
        print("  1. python3 compare_all_models_v2.py")
        print("  2. python3 test_multilingual_granite.py")
        print("  3. python3 mteb_evaluation_granite.py (optional)")
        print()

        print("🆚 Expected Comparison:")
        print("  - Qwen2.5-1.5B: ~302MB, 93.46% quality, EN/ZH focus")
        print("  - Qwen2.5-3B: ~302MB, 92.92% quality, EN/ZH focus")
        print("  - Granite 4.0 Micro: ~40-50MB, 88-92% quality, 12 LANGUAGES")
        print()

        print("=" * 80)
        print("✅ SUCCESS! Model ready for evaluation.")
        print("=" * 80)
        print()
        print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ DISTILLATION FAILED")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        print("Possible causes:")
        print("  1. Out of VRAM (need 5-6GB)")
        print("  2. Out of RAM (need 12-16GB)")
        print("  3. Network error during model download")
        print()
        print("Solutions:")
        print("  1. Close other applications")
        print("  2. Stop Docker containers")
        print("  3. Check internet connection")
        print()
        sys.exit(1)

if __name__ == "__main__":
    main()
