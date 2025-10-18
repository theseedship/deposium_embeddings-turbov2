#!/usr/bin/env python3
"""
Upload Qwen25-1024D Model2Vec to HuggingFace Hub

This will upload the distilled model to: tss-deposium/qwen25-deposium-1024d
"""

from huggingface_hub import HfApi, create_repo
import os
from pathlib import Path

# Configuration
REPO_ID = "tss-deposium/qwen25-deposium-1024d"
MODEL_PATH = "models/qwen25-deposium-1024d"

def upload_model():
    """Upload model to HuggingFace Hub"""

    # Check if model exists
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    print("=" * 80)
    print("🚀 Uploading Qwen25-1024D Model2Vec to HuggingFace Hub")
    print("=" * 80)
    print(f"Repository: {REPO_ID}")
    print(f"Model Path: {MODEL_PATH}")
    print()

    # Initialize HF API
    api = HfApi()

    # Check if logged in
    try:
        user_info = api.whoami()
        print(f"✅ Logged in as: {user_info['name']}")
    except Exception as e:
        print("❌ Not logged in to HuggingFace!")
        print("Please run: huggingface-cli login")
        return

    # Create repository if it doesn't exist
    print(f"\n📦 Creating repository: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print("✅ Repository created/verified")
    except Exception as e:
        print(f"⚠️  Repository creation: {e}")

    # Upload all files
    print(f"\n📤 Uploading model files...")

    files_to_upload = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "metadata.json",
        "modules.json",
        "README.md"
    ]

    for filename in files_to_upload:
        file_path = model_path / filename
        if file_path.exists():
            print(f"  Uploading {filename} ({file_path.stat().st_size / 1024 / 1024:.1f} MB)...")
            try:
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=filename,
                    repo_id=REPO_ID,
                    repo_type="model"
                )
                print(f"  ✅ {filename} uploaded")
            except Exception as e:
                print(f"  ❌ Failed to upload {filename}: {e}")
        else:
            print(f"  ⚠️  {filename} not found, skipping")

    print()
    print("=" * 80)
    print("🎉 Upload Complete!")
    print("=" * 80)
    print(f"Model available at: https://huggingface.co/{REPO_ID}")
    print()
    print("📊 Model Info:")
    print("  - Dimensions: 1024D")
    print("  - Size: ~65MB (298MB safetensors)")
    print("  - Quality: 0.841 overall")
    print("  - Instruction-Awareness: 0.953")
    print("  - Base: Qwen2.5-1.5B-Instruct (1.54B params distilled)")
    print()
    print("🚀 Railway will now be able to download this model at startup!")
    print()

if __name__ == "__main__":
    upload_model()
