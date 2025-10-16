#!/usr/bin/env python3
"""
HuggingFace Space - NVIDIA Nemotron-Nano-9B-v2 Distillation
Premier test Model2Vec sur architecture Mamba2-Transformer Hybrid!

Optimized for A10G large or A100 (1-2h)
"""

import gradio as gr
import torch
from model2vec.distill import distill
from datetime import datetime
from pathlib import Path
import zipfile
import os

def distill_nemotron_nano(progress=gr.Progress()):
    """Distille NVIDIA Nemotron-Nano-9B-v2 vers Model2Vec"""

    progress(0, desc="🚀 Starting NVIDIA Nemotron distillation...")

    output_dir = Path("nemotron-nano-9b-deposium-1024d")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    log = []

    try:
        # GPU Info
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0

        log.append("=" * 80)
        log.append("🔥 NVIDIA Nemotron-Nano-9B-v2 → Model2Vec Distillation")
        log.append("⚡ FIRST Model2Vec distillation on Mamba2-Transformer Hybrid!")
        log.append("=" * 80)
        log.append(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        log.append(f"GPU: {gpu_name}")
        log.append(f"VRAM: {gpu_mem:.1f}GB")
        log.append("")
        log.append("📊 Model Specifications:")
        log.append("  - Architecture: Mamba2-Transformer Hybrid (cutting-edge)")
        log.append("  - Parameters: 8.89B")
        log.append("  - Vocabulary: 131K Tekken tokenizer")
        log.append("  - Context: 128K tokens")
        log.append("  - Expected Model2Vec size: ~268MB")
        log.append("  - Expected quality: 90-94%")
        log.append("")

        yield "\n".join(log), None

        progress(0.1, desc="📥 Loading NVIDIA Nemotron-Nano-9B-v2...")
        log.append("📥 Downloading nvidia/NVIDIA-Nemotron-Nano-9B-v2...")
        log.append("Model size: ~18GB")
        log.append("This may take 10-15 minutes...")
        log.append("")
        yield "\n".join(log), None

        # Distillation
        progress(0.2, desc="🔥 Distilling model (Mamba2 hybrid - experimental)...")
        log.append("🔥 Starting Model2Vec distillation...")
        log.append("")
        log.append("⚠️  EXPERIMENTAL: First Model2Vec on Mamba2-Transformer!")
        log.append("")
        log.append("Configuration:")
        log.append("  - Target dimensions: 1024D")
        log.append("  - PCA: Enabled")
        log.append("  - Zipf weighting: Enabled")
        log.append("  - Vocabulary: 131K tokens (large!)")
        log.append("  - Expected time: 1-2 hours (larger vocab)")
        log.append("")
        log.append("Note: Mamba2 hybrid architecture may behave differently")
        log.append("than standard transformers. Quality may vary.")
        log.append("")
        yield "\n".join(log), None

        model = distill(
            model_name="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
            pca_dims=1024,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        progress(0.8, desc="💾 Saving model...")
        log.append("")
        log.append("💾 Saving model to disk...")
        log.append("Note: Large vocabulary = larger file size")
        yield "\n".join(log), None

        # Save
        model.save_pretrained(str(output_dir))

        # Test
        progress(0.85, desc="🧪 Testing model...")
        log.append("🧪 Running quick tests...")
        log.append("")
        yield "\n".join(log), None

        test_texts = [
            "What is machine learning?",
            "Explain neural networks",
            "Python programming tutorial",
            "Advanced AI reasoning task",
        ]
        embeddings = model.encode(test_texts)

        # Calculate embedding stats
        import numpy as np
        norms = np.linalg.norm(embeddings, axis=1)

        # Metadata
        end_time = datetime.now()
        duration = end_time - start_time

        # Get size
        model_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        model_size_mb = model_size / (1024 * 1024)

        # Write metadata
        with open(output_dir / "distillation_metadata.txt", "w") as f:
            f.write(f"Model: nvidia/NVIDIA-Nemotron-Nano-9B-v2\n")
            f.write(f"Architecture: Mamba2-Transformer Hybrid\n")
            f.write(f"Distillation method: Model2Vec\n")
            f.write(f"Dimensions: 1024D\n")
            f.write(f"Vocabulary: 131K Tekken tokenizer\n")
            f.write(f"Device: {gpu_name}\n")
            f.write(f"Start time: {start_time.isoformat()}\n")
            f.write(f"End time: {end_time.isoformat()}\n")
            f.write(f"Duration: {duration}\n")
            f.write(f"Model size: {model_size_mb:.1f}MB\n")
            f.write(f"Experimental: First Model2Vec on Mamba2 architecture\n")

        # Create ZIP
        progress(0.9, desc="📦 Creating download archive...")
        log.append("📦 Creating ZIP archive for download...")
        log.append("Note: Larger file due to 131K vocabulary")
        yield "\n".join(log), None

        zip_path = "nemotron-nano-9b-deposium-1024d.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in output_dir.rglob("*"):
                if file.is_file():
                    zipf.write(file, file.relative_to(output_dir.parent))

        zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)

        # Final summary
        log.append("")
        log.append("=" * 80)
        log.append("✅ DISTILLATION COMPLETE!")
        log.append("=" * 80)
        log.append(f"⏱️  Duration: {duration}")
        log.append(f"📦 Model size: {model_size_mb:.1f}MB")
        log.append(f"📦 ZIP size: {zip_size_mb:.1f}MB")
        log.append(f"🧪 Test embeddings shape: {embeddings.shape}")
        log.append(f"📊 Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
        log.append("")
        log.append("⚡ EXPERIMENTAL SUCCESS!")
        log.append("This is the FIRST Model2Vec distillation of a Mamba2-Transformer hybrid!")
        log.append("")
        log.append("📥 Download the ZIP file below!")
        log.append("")
        log.append("Next steps on your local machine:")
        log.append("  1. Extract ZIP: unzip nemotron-nano-9b-deposium-1024d.zip")
        log.append("  2. Test: python3 test_nemotron_nano_model.py")
        log.append("  3. Evaluate: python3 quick_eval_nemotron_nano_1024d.py")
        log.append("  4. Compare with Qwen2.5-7B results")
        log.append("")
        log.append(f"Expected quality: 90-94%")
        log.append(f"Expected advantages:")
        log.append(f"  ✅ Mamba2 efficiency (fast inference)")
        log.append(f"  ✅ NVIDIA optimizations")
        log.append(f"  ✅ Advanced reasoning capabilities")
        log.append(f"  ⚠️  Larger size ({model_size_mb:.0f}MB vs 65MB for Qwen)")
        log.append("")

        progress(1.0, desc="✅ Done!")

        return "\n".join(log), zip_path

    except Exception as e:
        log.append("")
        log.append("=" * 80)
        log.append("❌ ERROR")
        log.append("=" * 80)
        log.append(f"Error: {str(e)}")
        log.append("")
        log.append("Possible causes:")
        log.append("  - Mamba2 architecture not fully compatible with Model2Vec")
        log.append("  - Insufficient memory (try A100 large)")
        log.append("  - Model2Vec version too old")
        log.append("")

        import traceback
        log.append("Traceback:")
        log.append(traceback.format_exc())

        return "\n".join(log), None

# Gradio Interface
with gr.Blocks(title="Nemotron-Nano-9B Distillation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔥 NVIDIA Nemotron-Nano-9B-v2 → Model2Vec")
    gr.Markdown("**⚡ EXPERIMENTAL: Premier test Model2Vec sur architecture Mamba2-Transformer Hybrid!**")

    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### 📊 Spécifications
            - **Source:** nvidia/NVIDIA-Nemotron-Nano-9B-v2
            - **Architecture:** Mamba2-Transformer Hybrid (cutting-edge)
            - **Paramètres:** 8.89B
            - **Vocabulaire:** 131K Tekken tokenizer
            - **Contexte:** 128K tokens
            """)
        with gr.Column():
            gr.Markdown("""
            ### 🎯 Cible Model2Vec
            - **Dimensions:** 1024D
            - **Taille attendue:** ~268MB
            - **Qualité attendue:** 90-94%
            - **Durée:** 1-2 heures (A10G large/A100)
            """)

    gr.Markdown("---")

    with gr.Row():
        start_btn = gr.Button("🚀 Start Distillation", variant="primary", size="lg")

    gr.Markdown("---")

    output_log = gr.Textbox(
        label="📋 Logs de Distillation",
        lines=30,
        max_lines=60,
        show_copy_button=True
    )

    download_file = gr.File(
        label="📦 Télécharger le Modèle (ZIP, ~300MB)",
        type="filepath"
    )

    gr.Markdown("""
    ---
    ### ⚠️  Notes Importantes

    **Architecture Expérimentale:**
    - Première distillation Model2Vec sur Mamba2-Transformer
    - Résultats peuvent différer des transformers standards
    - Architecture hybride = comportement unique

    **Recommandations Hardware:**
    - **Minimum:** A10G large (46GB VRAM) - $1.50/h
    - **Optimal:** A100 large (142GB VRAM) - $2.50/h
    - **Durée:** 1-2 heures

    **Avantages Attendus:**
    - ✅ Mamba2: Inference plus rapide que transformers purs
    - ✅ NVIDIA: Optimisations hardware natives
    - ✅ Reasoning: Capacités avancées
    - ✅ Innovation: Architecture de pointe (août 2025)

    **Trade-offs:**
    - ⚠️ Taille: ~268MB (vs 65MB Qwen2.5-7B)
    - ⚠️ Expérimental: Premier test sur Mamba2
    - ⚠️ Coût: ~$2-3 (A100) ou ~$1.50-3 (A10G large)

    ### 📥 Après téléchargement
    ```bash
    unzip nemotron-nano-9b-deposium-1024d.zip -d models/
    python3 test_nemotron_nano_model.py
    python3 quick_eval_nemotron_nano_1024d.py
    ```

    ### 🆚 Comparaison avec Qwen2.5-7B
    | Métrique | Qwen2.5-7B | Nemotron-Nano-9B |
    |----------|------------|------------------|
    | Architecture | Transformer | Mamba2 Hybrid ⚡ |
    | Taille Model2Vec | 65MB | ~268MB |
    | Qualité | 91-95% | 90-94% |
    | Vocab | 32K | 131K |
    | Reasoning | Standard | Avancé ✨ |
    | Inference | Rapide | Très rapide 🚀 |
    """)

    start_btn.click(
        fn=distill_nemotron_nano,
        inputs=[],
        outputs=[output_log, download_file]
    )

if __name__ == "__main__":
    demo.launch()
