#!/usr/bin/env python3
"""
HuggingFace Space - Qwen2.5-7B Distillation
Optimized for A10G small GPU (30-60 min)
"""

import gradio as gr
import torch
from model2vec import distill_model
from datetime import datetime
from pathlib import Path
import zipfile
import os

def distill_qwen25_7b(progress=gr.Progress()):
    """Distille Qwen2.5-7B vers Model2Vec"""

    progress(0, desc="🚀 Starting distillation...")

    output_dir = Path("qwen25-7b-deposium-1024d")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    log = []

    try:
        # GPU Info
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0

        log.append("=" * 80)
        log.append("🚀 Qwen2.5-7B → Model2Vec Distillation")
        log.append("=" * 80)
        log.append(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        log.append(f"GPU: {gpu_name}")
        log.append(f"VRAM: {gpu_mem:.1f}GB")
        log.append("")

        yield "\n".join(log), None

        progress(0.1, desc="📥 Loading Qwen2.5-7B (14GB download)...")
        log.append("📥 Downloading Qwen2.5-7B-Instruct (~14GB)...")
        log.append("This may take 5-10 minutes...")
        log.append("")
        yield "\n".join(log), None

        # Distillation
        progress(0.2, desc="🔥 Distilling model (main process)...")
        log.append("🔥 Starting Model2Vec distillation...")
        log.append("Configuration:")
        log.append("  - Dimensions: 1024D")
        log.append("  - PCA: Enabled")
        log.append("  - Zipf weighting: Enabled")
        log.append("  - Expected time: 30-60 minutes")
        log.append("")
        yield "\n".join(log), None

        model = distill_model(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            pca_dims=1024,
            apply_pca=True,
            use_subword=True,
            apply_zipf=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            show_progress_bar=True,
        )

        progress(0.8, desc="💾 Saving model...")
        log.append("💾 Saving model to disk...")
        yield "\n".join(log), None

        # Save
        model.save_pretrained(str(output_dir))

        # Test
        progress(0.85, desc="🧪 Testing model...")
        log.append("🧪 Running quick tests...")
        yield "\n".join(log), None

        test_texts = [
            "What is machine learning?",
            "Explain neural networks",
            "Python programming tutorial",
        ]
        embeddings = model.encode(test_texts, show_progress_bar=False)

        # Metadata
        end_time = datetime.now()
        duration = end_time - start_time

        # Get size
        model_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        model_size_mb = model_size / (1024 * 1024)

        # Write metadata
        with open(output_dir / "distillation_metadata.txt", "w") as f:
            f.write(f"Model: Qwen/Qwen2.5-7B-Instruct\n")
            f.write(f"Distillation method: Model2Vec\n")
            f.write(f"Dimensions: 1024D\n")
            f.write(f"Device: {gpu_name}\n")
            f.write(f"Start time: {start_time.isoformat()}\n")
            f.write(f"End time: {end_time.isoformat()}\n")
            f.write(f"Duration: {duration}\n")
            f.write(f"Model size: {model_size_mb:.1f}MB\n")

        # Create ZIP
        progress(0.9, desc="📦 Creating download archive...")
        log.append("📦 Creating ZIP archive for download...")
        yield "\n".join(log), None

        zip_path = "qwen25-7b-deposium-1024d.zip"
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
        log.append("")
        log.append("📥 Download the ZIP file below!")
        log.append("")
        log.append("Next steps on your local machine:")
        log.append("  1. Extract ZIP: unzip qwen25-7b-deposium-1024d.zip")
        log.append("  2. Test: python3 test_qwen25_7b_model.py")
        log.append("  3. Evaluate: python3 quick_eval_qwen25_7b_1024d.py")
        log.append("  4. If score ≥ 91%, deploy!")
        log.append("")
        log.append(f"Expected quality: 91-95%")
        log.append(f"Expected instruction-awareness: 96-98%")
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

        import traceback
        log.append("Traceback:")
        log.append(traceback.format_exc())

        return "\n".join(log), None

# Gradio Interface
with gr.Blocks(title="Qwen2.5-7B Distillation") as demo:
    gr.Markdown("# 🚀 Qwen2.5-7B → Model2Vec Distillation")
    gr.Markdown("Distille **Qwen/Qwen2.5-7B-Instruct** en **Model2Vec 1024D** (~65MB)")

    with gr.Row():
        gr.Markdown("""
        ### Spécifications
        - **Source:** Qwen2.5-7B-Instruct (14GB)
        - **Cible:** Model2Vec 1024D (~65MB)
        - **Qualité attendue:** 91-95%
        - **Durée:** 30-60 minutes sur A10G
        """)

    with gr.Row():
        start_btn = gr.Button("🚀 Start Distillation", variant="primary", size="lg")

    gr.Markdown("---")

    output_log = gr.Textbox(
        label="📋 Logs de Distillation",
        lines=25,
        max_lines=50,
        show_copy_button=True
    )

    download_file = gr.File(
        label="📦 Télécharger le Modèle (ZIP)",
        type="filepath"
    )

    gr.Markdown("""
    ---
    ### Instructions
    1. Cliquez sur "Start Distillation"
    2. Attendez 30-60 minutes
    3. Téléchargez le fichier ZIP
    4. Extrayez et testez sur votre machine locale

    ### Après téléchargement
    ```bash
    unzip qwen25-7b-deposium-1024d.zip -d models/
    python3 test_qwen25_7b_model.py
    python3 quick_eval_qwen25_7b_1024d.py
    ```
    """)

    start_btn.click(
        fn=distill_qwen25_7b,
        inputs=[],
        outputs=[output_log, download_file]
    )

if __name__ == "__main__":
    demo.launch()
