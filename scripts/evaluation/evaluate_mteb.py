#!/usr/bin/env python3
"""
MTEB Evaluation for LEAF Model
Tests on STS Benchmark and STS22 to compare with base model
"""
import torch
from transformers import AutoTokenizer
from pathlib import Path
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Wrapper class for MTEB compatibility
class LEAFModelWrapper:
    """Wrapper to make LEAF model compatible with MTEB"""

    def __init__(self, model_path: str):
        logger.info(f"Loading LEAF model from {model_path}")

        # Load model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model = checkpoint['model']
        self.model.eval()

        # Load tokenizer
        tokenizer_path = Path(model_path).parent
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        self.model.set_tokenizer(self.tokenizer)

        logger.info("✅ LEAF model loaded!")

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Encode sentences to embeddings (MTEB-compatible interface)
        """
        all_embeddings = []

        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            with torch.no_grad():
                embeddings = self.model.encode(
                    batch,
                    device='cpu',
                    normalize=normalize_embeddings
                )

            all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)

        return all_embeddings


def main():
    """Run MTEB evaluation on LEAF model"""

    try:
        # Import MTEB
        from mteb import MTEB
        logger.info("✅ MTEB imported successfully")
    except ImportError:
        logger.error("❌ MTEB not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "mteb"])
        from mteb import MTEB

    # Load LEAF model
    model_path = "models/leaf_cpu/model_quantized.pt"
    if not Path(model_path).exists():
        logger.error(f"❌ Model not found at {model_path}")
        logger.info("Please ensure the model is in the correct location")
        return

    model = LEAFModelWrapper(model_path)

    # Define tasks to evaluate (same as in training config)
    tasks = [
        "STSBenchmark",
        "STS22"  # Multilingual STS
    ]

    logger.info(f"📊 Running MTEB evaluation on tasks: {tasks}")

    # Run evaluation
    evaluation = MTEB(tasks=tasks)
    results = evaluation.run(
        model,
        output_folder="mteb_results",
        eval_splits=["test"]
    )

    # Print results
    logger.info("\n" + "="*60)
    logger.info("📊 MTEB EVALUATION RESULTS")
    logger.info("="*60)

    for task_name, task_results in results.items():
        logger.info(f"\n🎯 {task_name}:")

        if "test" in task_results:
            test_results = task_results["test"]

            # Main metrics
            if "cos_sim" in test_results:
                cos_sim = test_results["cos_sim"]
                if "spearman" in cos_sim:
                    logger.info(f"  Spearman: {cos_sim['spearman']:.4f}")
                if "pearson" in cos_sim:
                    logger.info(f"  Pearson: {cos_sim['pearson']:.4f}")

            # Print all available metrics
            logger.info(f"  All metrics: {test_results}")

    logger.info("\n" + "="*60)
    logger.info(f"✅ Results saved to: mteb_results/")
    logger.info("="*60)

    # Save summary
    summary_file = Path("mteb_results/summary.txt")
    summary_file.parent.mkdir(exist_ok=True)

    with open(summary_file, "w") as f:
        f.write("MTEB Evaluation Summary - LEAF Model (512 tokens)\n")
        f.write("="*60 + "\n\n")

        for task_name, task_results in results.items():
            f.write(f"{task_name}:\n")
            if "test" in task_results:
                test_results = task_results["test"]
                if "cos_sim" in test_results:
                    cos_sim = test_results["cos_sim"]
                    if "spearman" in cos_sim:
                        f.write(f"  Spearman: {cos_sim['spearman']:.4f}\n")
                    if "pearson" in cos_sim:
                        f.write(f"  Pearson: {cos_sim['pearson']:.4f}\n")
            f.write("\n")

        f.write("\nComparison with EmbeddingGemma-300m:\n")
        f.write("  Base model (768D): ~61.15 MTEB score\n")
        f.write("  LEAF (512 tokens): See above\n")

    logger.info(f"📄 Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
