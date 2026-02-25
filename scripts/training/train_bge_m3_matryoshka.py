#!/usr/bin/env python3
"""
BGE-M3 Matryoshka Fine-tuning + ONNX INT8 Export
=================================================

Fine-tunes BAAI/bge-m3 with MatryoshkaLoss for truncatable embeddings [1024, 768, 512, 256],
then exports to ONNX with INT8 dynamic quantization.

Designed for Google Colab (free T4 16GB) or any GPU with >= 16GB VRAM.

Usage (Colab):
    1. Upload this script or copy-paste cells
    2. Runtime > Change runtime type > T4 GPU
    3. Run all cells

Usage (local/cloud):
    pip install -r requirements-training.txt
    python scripts/training/train_bge_m3_matryoshka.py

Output:
    - Fine-tuned model: ./bge-m3-matryoshka-deposium/
    - ONNX INT8 model: ./bge-m3-matryoshka-onnx-int8/
    - Pushed to HuggingFace Hub: tss-deposium/bge-m3-matryoshka-1024d

Reference: docs/2026/r&d/bge-matryoshka.md
"""

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config - adjust these before running
# ---------------------------------------------------------------------------

# HuggingFace Hub
HF_REPO_ID = os.getenv("HF_REPO_ID", "tss-deposium/bge-m3-matryoshka-1024d")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # set via env or Colab secrets

# Base model
BASE_MODEL = "BAAI/bge-m3"

# Matryoshka dimensions (must include native 1024)
MATRYOSHKA_DIMS = [1024, 768, 512, 256]

# Training hyperparameters
NUM_EPOCHS = 4
BATCH_SIZE = 16  # T4 16GB handles 16 comfortably, reduce to 8 if OOM
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

# Dataset config
# Uses sentence-transformers official datasets (no auth required, always available)
ALLNLI_MAX_SAMPLES = 0  # 0 = use all (~560k pairs)
NQ_MAX_SAMPLES = 0  # 0 = use all (~100k pairs)
GOOAQ_MAX_SAMPLES = 50_000  # cap from GooAQ (3M+ total)

# Output dirs
OUTPUT_DIR = "./bge-m3-matryoshka-deposium"
ONNX_OUTPUT_DIR = "./bge-m3-matryoshka-onnx-int8"

# Eval
EVAL_SPLIT_RATIO = 0.05  # 5% for evaluation


# ---------------------------------------------------------------------------
# Step 0: Install dependencies (for Colab)
# ---------------------------------------------------------------------------

def install_dependencies():
    """Install required packages. Safe to run multiple times."""
    import subprocess

    packages = [
        "sentence-transformers>=3.3.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        "optimum[onnxruntime]>=1.22.0",
        "onnxruntime>=1.19.0",
        "huggingface_hub>=0.25.0",
    ]

    logger.info("Installing dependencies...")
    for pkg in packages:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    logger.info("Dependencies installed.")


# ---------------------------------------------------------------------------
# Step 1: Load & prepare dataset
# ---------------------------------------------------------------------------

def load_allnli_pairs(max_samples: int) -> list[dict]:
    """Load pairs from sentence-transformers/all-nli (NLI triplets, ~560k)."""
    from datasets import load_dataset

    pairs = []
    try:
        logger.info("Loading sentence-transformers/all-nli...")
        ds = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
        for row in ds:
            pairs.append({"anchor": row["anchor"], "positive": row["positive"]})
            if max_samples and len(pairs) >= max_samples:
                break
        logger.info(f"  all-nli: {len(pairs)} pairs")
    except Exception as e:
        logger.warning(f"  Skipping all-nli: {e}")

    return pairs


def load_nq_pairs(max_samples: int) -> list[dict]:
    """Load pairs from sentence-transformers/natural-questions (~100k Q&A)."""
    from datasets import load_dataset

    pairs = []
    try:
        logger.info("Loading sentence-transformers/natural-questions...")
        ds = load_dataset("sentence-transformers/natural-questions", split="train")
        for row in ds:
            pairs.append({"anchor": row["query"], "positive": row["answer"]})
            if max_samples and len(pairs) >= max_samples:
                break
        logger.info(f"  natural-questions: {len(pairs)} pairs")
    except Exception as e:
        logger.warning(f"  Skipping natural-questions: {e}")

    return pairs


def load_gooaq_pairs(max_samples: int) -> list[dict]:
    """Load pairs from sentence-transformers/gooaq (3M+ Q&A, streamed)."""
    from datasets import load_dataset

    pairs = []
    try:
        logger.info(f"Loading sentence-transformers/gooaq (max {max_samples})...")
        ds = load_dataset("sentence-transformers/gooaq", split="train", streaming=True)
        for row in ds:
            pairs.append({"anchor": row["question"], "positive": row["answer"]})
            if len(pairs) >= max_samples:
                break
        logger.info(f"  gooaq: {len(pairs)} pairs")
    except Exception as e:
        logger.warning(f"  Skipping gooaq: {e}")

    return pairs


def prepare_dataset():
    """Combine all-nli + natural-questions + gooaq into train/eval splits."""
    from datasets import Dataset

    logger.info("=" * 60)
    logger.info("STEP 1: Preparing dataset")
    logger.info("=" * 60)

    # Load from sentence-transformers official datasets (no auth required)
    nli_pairs = load_allnli_pairs(ALLNLI_MAX_SAMPLES)
    nq_pairs = load_nq_pairs(NQ_MAX_SAMPLES)
    gooaq_pairs = load_gooaq_pairs(GOOAQ_MAX_SAMPLES)

    all_pairs = nli_pairs + nq_pairs + gooaq_pairs
    logger.info(f"Total pairs: {len(all_pairs)} "
                f"(all-nli: {len(nli_pairs)}, nq: {len(nq_pairs)}, gooaq: {len(gooaq_pairs)})")

    if len(all_pairs) < 1000:
        logger.error("Not enough data! Need at least 1000 pairs. Check dataset loading.")
        sys.exit(1)

    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(all_pairs)

    split_idx = int(len(all_pairs) * (1 - EVAL_SPLIT_RATIO))
    train_pairs = all_pairs[:split_idx]
    eval_pairs = all_pairs[split_idx:]

    train_dataset = Dataset.from_list(train_pairs)
    eval_dataset = Dataset.from_list(eval_pairs)

    logger.info(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    return train_dataset, eval_dataset


# ---------------------------------------------------------------------------
# Step 2: Fine-tune with MatryoshkaLoss
# ---------------------------------------------------------------------------

def train_model(train_dataset, eval_dataset):
    """Fine-tune BGE-M3 with MatryoshkaLoss."""
    import torch
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss

    logger.info("=" * 60)
    logger.info("STEP 2: Fine-tuning BGE-M3 with MatryoshkaLoss")
    logger.info("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        logger.warning("No GPU detected! Training will be very slow.")

    # Load base model
    logger.info(f"Loading {BASE_MODEL}...")
    model = SentenceTransformer(BASE_MODEL)
    logger.info(f"Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Configure loss
    inner_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(
        model,
        inner_loss,
        matryoshka_dims=MATRYOSHKA_DIMS,
        matryoshka_weights=[1, 1, 1, 1],  # equal weights for all dimensions
    )

    # Adjust batch size based on available VRAM
    effective_batch_size = BATCH_SIZE
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_mem < 16:
            effective_batch_size = 8
            logger.info(f"Reduced batch size to {effective_batch_size} for {gpu_mem:.0f}GB GPU")

    # Training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=effective_batch_size,
        per_device_eval_batch_size=effective_batch_size,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final model
    model.save_pretrained(OUTPUT_DIR)
    logger.info(f"Model saved to {OUTPUT_DIR}")

    return model


# ---------------------------------------------------------------------------
# Step 3: Export ONNX + INT8 quantization
# ---------------------------------------------------------------------------

def export_onnx_int8(model=None):
    """Export to ONNX with INT8 dynamic quantization."""
    from sentence_transformers import SentenceTransformer

    logger.info("=" * 60)
    logger.info("STEP 3: Exporting to ONNX INT8")
    logger.info("=" * 60)

    if model is None:
        logger.info(f"Loading model from {OUTPUT_DIR}...")
        model = SentenceTransformer(OUTPUT_DIR)

    # Method: sentence-transformers native export
    try:
        from sentence_transformers import export_dynamic_quantized_onnx_model

        logger.info("Using sentence-transformers native ONNX export...")
        export_dynamic_quantized_onnx_model(
            model,
            quantization_config="avx512_vnni",
            model_name_or_path=ONNX_OUTPUT_DIR,
        )
        logger.info(f"ONNX INT8 model saved to {ONNX_OUTPUT_DIR}")
        return

    except (ImportError, Exception) as e:
        logger.warning(f"Native export failed ({e}), falling back to optimum...")

    # Fallback: HuggingFace Optimum
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    logger.info("Exporting with HuggingFace Optimum...")

    # Step 3a: Export to ONNX FP32
    onnx_fp32_dir = f"{ONNX_OUTPUT_DIR}-fp32-tmp"
    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        OUTPUT_DIR, export=True
    )
    ort_model.save_pretrained(onnx_fp32_dir)
    logger.info(f"ONNX FP32 exported to {onnx_fp32_dir}")

    # Step 3b: Quantize to INT8
    quantizer = ORTQuantizer.from_pretrained(onnx_fp32_dir)
    qconfig = AutoQuantizationConfig.avx512_vnni(
        is_static=False, per_channel=True
    )

    Path(ONNX_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    quantizer.quantize(
        save_dir=ONNX_OUTPUT_DIR,
        quantization_config=qconfig,
    )
    logger.info(f"ONNX INT8 model saved to {ONNX_OUTPUT_DIR}")

    # Copy tokenizer files
    import shutil
    for f in Path(onnx_fp32_dir).glob("*.json"):
        if "model" not in f.name.lower() or f.name == "config.json":
            shutil.copy2(f, ONNX_OUTPUT_DIR)

    # Cleanup tmp
    shutil.rmtree(onnx_fp32_dir, ignore_errors=True)
    logger.info("Cleanup done.")


# ---------------------------------------------------------------------------
# Step 4: Validate embeddings
# ---------------------------------------------------------------------------

def validate_model():
    """Quick validation: compare dimensions and cosine similarity."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    logger.info("=" * 60)
    logger.info("STEP 4: Validation")
    logger.info("=" * 60)

    # Load the fine-tuned model (PyTorch)
    model = SentenceTransformer(OUTPUT_DIR)

    test_sentences = [
        "Comment résilier un contrat d'assurance ?",
        "La résiliation d'un contrat peut être effectuée par lettre recommandée avec accusé de réception.",
        "Les prévisions météo annoncent de la pluie demain.",
        "How to cancel an insurance contract?",
        "El contrato puede ser rescindido mediante carta certificada.",
    ]

    # Encode at full dimension
    embeddings_full = model.encode(test_sentences, normalize_embeddings=True)

    # Test Matryoshka: truncate and measure quality
    logger.info("\nMatryoshka dimension comparison:")
    logger.info(f"{'Dim':>6} | {'Sim(q,doc)':>10} | {'Sim(q,noise)':>12} | {'Delta':>8} | {'Status':>8}")
    logger.info("-" * 60)

    for dim in MATRYOSHKA_DIMS:
        embs = embeddings_full[:, :dim]
        # Re-normalize after truncation
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / norms

        # Similarity: query vs relevant doc (should be high)
        sim_relevant = np.dot(embs[0], embs[1])
        # Similarity: query vs noise (should be low)
        sim_noise = np.dot(embs[0], embs[2])
        delta = sim_relevant - sim_noise

        status = "OK" if delta > 0.1 else "WARN"
        logger.info(f"{dim:>6} | {sim_relevant:>10.4f} | {sim_noise:>12.4f} | {delta:>8.4f} | {status:>8}")

    # Cross-lingual test
    logger.info("\nCross-lingual similarity (FR query vs translations):")
    for i, (lang, idx) in enumerate([("FR doc", 1), ("EN query", 3), ("ES doc", 4)]):
        for dim in [1024, 256]:
            embs = embeddings_full[:, :dim]
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / norms
            sim = np.dot(embs[0], embs[idx])
            logger.info(f"  {lang} @ {dim}D: {sim:.4f}")

    logger.info("\nValidation complete.")
    return model


# ---------------------------------------------------------------------------
# Step 5: Push to HuggingFace Hub
# ---------------------------------------------------------------------------

def push_to_hub(model=None):
    """Push both PyTorch and ONNX models to HuggingFace Hub."""
    from huggingface_hub import HfApi, login

    logger.info("=" * 60)
    logger.info("STEP 5: Pushing to HuggingFace Hub")
    logger.info("=" * 60)

    token = HF_TOKEN
    if not token:
        # Try Colab secrets
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            pass

    if not token:
        logger.warning("No HF_TOKEN found. Skipping Hub push.")
        logger.info(f"To push manually later:\n"
                     f"  huggingface-cli login\n"
                     f"  huggingface-cli upload {HF_REPO_ID} {OUTPUT_DIR}\n"
                     f"  huggingface-cli upload {HF_REPO_ID}-onnx-int8 {ONNX_OUTPUT_DIR}")
        return

    # Pass token directly to HfApi instead of login() to avoid auth issues
    api = HfApi(token=token)

    # Push PyTorch model
    logger.info(f"Pushing PyTorch model to {HF_REPO_ID}...")
    api.create_repo(HF_REPO_ID, exist_ok=True, private=False)
    api.upload_folder(
        folder_path=OUTPUT_DIR,
        repo_id=HF_REPO_ID,
        commit_message="BGE-M3 Matryoshka fine-tuned [1024, 768, 512, 256]",
    )
    logger.info(f"PyTorch model pushed: https://huggingface.co/{HF_REPO_ID}")

    # Push ONNX INT8 model
    onnx_repo_id = f"{HF_REPO_ID}-onnx-int8"
    if Path(ONNX_OUTPUT_DIR).exists():
        logger.info(f"Pushing ONNX INT8 model to {onnx_repo_id}...")
        api.create_repo(onnx_repo_id, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=ONNX_OUTPUT_DIR,
            repo_id=onnx_repo_id,
            commit_message="BGE-M3 Matryoshka ONNX INT8 [1024, 768, 512, 256]",
        )
        logger.info(f"ONNX model pushed: https://huggingface.co/{onnx_repo_id}")

    logger.info("All models pushed to Hub!")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("BGE-M3 Matryoshka Fine-tuning Pipeline")
    logger.info("=" * 60)
    logger.info(f"Base model:    {BASE_MODEL}")
    logger.info(f"Dimensions:    {MATRYOSHKA_DIMS}")
    logger.info(f"Epochs:        {NUM_EPOCHS}")
    logger.info(f"Batch size:    {BATCH_SIZE}")
    logger.info(f"Hub repo:      {HF_REPO_ID}")
    logger.info("")

    # Step 0: Install deps (Colab)
    if "google.colab" in sys.modules or os.getenv("COLAB_GPU"):
        install_dependencies()

    # Step 1: Dataset
    train_dataset, eval_dataset = prepare_dataset()

    # Step 2: Train
    model = train_model(train_dataset, eval_dataset)

    # Step 3: Export ONNX INT8
    export_onnx_int8(model)

    # Step 4: Validate
    validate_model()

    # Step 5: Push to Hub
    push_to_hub(model)

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"PyTorch model: {OUTPUT_DIR}")
    logger.info(f"ONNX INT8:     {ONNX_OUTPUT_DIR}")
    logger.info(f"Hub:           https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
