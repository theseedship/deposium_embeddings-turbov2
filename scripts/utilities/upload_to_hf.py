#!/usr/bin/env python3
"""
Upload Model2Vec models to HuggingFace Hub

Configure via environment variables:
- HF_UPLOAD_REPO_ID: Target repository (default: tss-deposium/m2v-bge-m3-1024d)
- HF_UPLOAD_MODEL_PATH: Local model path (default: models/m2v-bge-m3-1024d)
"""

import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path

# Configuration from environment variables
REPO_ID = os.getenv("HF_UPLOAD_REPO_ID", "tss-deposium/m2v-bge-m3-1024d")
MODEL_PATH = os.getenv("HF_UPLOAD_MODEL_PATH", "models/m2v-bge-m3-1024d")


def upload_model():
    """Upload model to HuggingFace Hub"""

    # Check if model exists
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model_name = model_path.name

    print("=" * 80)
    print(f"Uploading {model_name} to HuggingFace Hub")
    print("=" * 80)
    print(f"Repository: {REPO_ID}")
    print(f"Model Path: {MODEL_PATH}")
    print()

    # Initialize HF API
    api = HfApi()

    # Check if logged in
    try:
        user_info = api.whoami()
        print(f"Logged in as: {user_info['name']}")
    except Exception as e:
        print("Not logged in to HuggingFace!")
        print("Please run: huggingface-cli login")
        return

    # Create repository if it doesn't exist
    print(f"\nCreating repository: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print("Repository created/verified")
    except Exception as e:
        print(f"Repository creation: {e}")

    # Upload all files
    print(f"\nUploading model files...")

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
                print(f"  {filename} uploaded")
            except Exception as e:
                print(f"  Failed to upload {filename}: {e}")
        else:
            print(f"  {filename} not found, skipping")

    print()
    print("=" * 80)
    print("Upload Complete!")
    print("=" * 80)
    print(f"Model available at: https://huggingface.co/{REPO_ID}")
    print()


if __name__ == "__main__":
    upload_model()
