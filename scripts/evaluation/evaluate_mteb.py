#!/usr/bin/env python3
"""
MTEB Evaluation for Deposium Embeddings (v11.0.0)
Tests current models: m2v-bge-m3-1024d, bge-m3-onnx, gemma-768d
"""
import numpy as np
from pathlib import Path
from typing import List
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model2VecWrapper:
    """Wrapper to make Model2Vec models compatible with MTEB"""

    def __init__(self, model_name: str = "m2v-bge-m3-1024d"):
        from model2vec import StaticModel

        # Model mapping to HuggingFace IDs
        HF_MODELS = {
            "m2v-bge-m3-1024d": os.getenv("HF_MODEL_M2V_BGE_M3", "tss-deposium/m2v-bge-m3-1024d"),
            "gemma-768d": os.getenv("HF_MODEL_GEMMA_768D", "tss-deposium/gemma-deposium-768d"),
        }

        hub_id = HF_MODELS.get(model_name)
        if not hub_id:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(HF_MODELS.keys())}")

        logger.info(f"Loading {model_name} from {hub_id}")
        self.model = StaticModel.from_pretrained(hub_id)
        self.model_name = model_name
        logger.info(f"Loaded {model_name}")

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode sentences to embeddings (MTEB-compatible interface)"""
        embeddings = self.model.encode(sentences)

        if normalize_embeddings:
            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-9)

        return embeddings


class OnnxWrapper:
    """Wrapper for ONNX models (bge-m3-onnx)"""

    def __init__(self, model_name: str = "bge-m3-onnx"):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from huggingface_hub import hf_hub_download

        hub_id = os.getenv("HF_MODEL_BGE_M3_ONNX", "gpahal/bge-m3-onnx-int8")
        logger.info(f"Loading {model_name} from {hub_id}")

        # Download model
        model_path = hf_hub_download(repo_id=hub_id, filename="model_quantized.onnx")

        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.tokenizer = AutoTokenizer.from_pretrained(hub_id)
        self.model_name = model_name
        logger.info(f"Loaded {model_name}")

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode sentences to embeddings (MTEB-compatible interface)"""
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=8192, return_tensors="np")

            outputs = self.session.run(
                ["dense_vecs"],
                {"input_ids": inputs["input_ids"].astype(np.int64), "attention_mask": inputs["attention_mask"].astype(np.int64)}
            )
            all_embeddings.append(outputs[0])

        embeddings = np.vstack(all_embeddings)

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-9)

        return embeddings


def main():
    """Run MTEB evaluation on current Deposium models"""
    import argparse

    parser = argparse.ArgumentParser(description="MTEB Evaluation for Deposium Embeddings")
    parser.add_argument("--model", default="m2v-bge-m3-1024d",
                       choices=["m2v-bge-m3-1024d", "bge-m3-onnx", "gemma-768d"],
                       help="Model to evaluate")
    parser.add_argument("--tasks", nargs="+", default=["STSBenchmark", "STS22"],
                       help="MTEB tasks to run")
    args = parser.parse_args()

    try:
        from mteb import MTEB
        logger.info("MTEB imported successfully")
    except ImportError:
        logger.error("MTEB not installed. Install with: pip install mteb")
        return

    # Load model
    if args.model in ["m2v-bge-m3-1024d", "gemma-768d"]:
        model = Model2VecWrapper(args.model)
    elif args.model == "bge-m3-onnx":
        model = OnnxWrapper(args.model)
    else:
        logger.error(f"Unknown model: {args.model}")
        return

    logger.info(f"Running MTEB evaluation on tasks: {args.tasks}")

    # Run evaluation
    evaluation = MTEB(tasks=args.tasks)
    results = evaluation.run(
        model,
        output_folder=f"mteb_results/{args.model}",
        eval_splits=["test"]
    )

    # Print results
    logger.info("\n" + "="*60)
    logger.info(f"MTEB EVALUATION RESULTS - {args.model}")
    logger.info("="*60)

    for task_name, task_results in results.items():
        logger.info(f"\n{task_name}:")
        if "test" in task_results:
            test_results = task_results["test"]
            if "cos_sim" in test_results:
                cos_sim = test_results["cos_sim"]
                if "spearman" in cos_sim:
                    logger.info(f"  Spearman: {cos_sim['spearman']:.4f}")
                if "pearson" in cos_sim:
                    logger.info(f"  Pearson: {cos_sim['pearson']:.4f}")

    logger.info(f"\nResults saved to: mteb_results/{args.model}/")


if __name__ == "__main__":
    main()
