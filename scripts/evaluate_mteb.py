"""
MTEB Evaluation Script for Model2Vec Models
Lightweight evaluation for Railway deployment

Usage:
    python scripts/evaluate_mteb.py --model gemma-768d
    python scripts/evaluate_mteb.py --model qwen3-m2v --tasks classification
    python scripts/evaluate_mteb.py --all --quick
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from mteb import MTEB
    import mteb
except ImportError:
    print("❌ MTEB not installed. Install with: pip install mteb")
    print("   Note: MTEB is heavy (~500MB). Only install for evaluation, not production.")
    sys.exit(1)

from model2vec import StaticModel
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model2VecWrapper:
    """Wrapper to make Model2Vec compatible with MTEB"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f"Loading {model_name}...")
        self.model = StaticModel.from_pretrained(model_name)
        logger.info(f"✅ {model_name} loaded")

    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        """Encode sentences to embeddings"""
        return self.model.encode(sentences, show_progress_bar=kwargs.get('show_progress_bar', False))


# Available models for testing
MODELS = {
    "gemma-768d": "tss-deposium/gemma-deposium-768d",
    "qwen3-m2v": "Pringled/m2v-Qwen3-Embedding-0.6B",
}

# Lightweight tasks for quick evaluation
QUICK_TASKS = [
    "Banking77Classification",  # Classification
    "ToxicConversationsClassification",  # Binary classification
    "EmotionClassification",  # Multi-class
]

# Reranking-specific tasks
RERANKING_TASKS = [
    "AskUbuntuDupQuestions",  # Reranking
    "StackOverflowDupQuestions",  # Reranking
]

# Full MTEB task categories (for comprehensive evaluation)
TASK_CATEGORIES = {
    "classification": ["Banking77Classification", "EmotionClassification", "ToxicConversationsClassification"],
    "clustering": ["ArxivClusteringP2P", "BiorxivClusteringP2P"],
    "reranking": ["AskUbuntuDupQuestions", "StackOverflowDupQuestions"],
    "retrieval": ["NFCorpus", "SciFact"],  # Small retrieval tasks
    "sts": ["STS12", "STS13"],  # Semantic Textual Similarity
}


def evaluate_model(model_key: str, tasks: List[str] = None, quick: bool = False):
    """
    Evaluate a Model2Vec model on MTEB tasks

    Args:
        model_key: Key from MODELS dict (e.g., "gemma-768d")
        tasks: List of task names to evaluate, or None for all
        quick: If True, run only quick lightweight tasks
    """
    if model_key not in MODELS:
        logger.error(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
        return

    model_name = MODELS[model_key]
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 Evaluating: {model_key} ({model_name})")
    logger.info(f"{'='*60}\n")

    # Load model
    model_wrapper = Model2VecWrapper(model_name)

    # Determine tasks to run
    if quick:
        task_list = QUICK_TASKS
        logger.info(f"⚡ Quick mode: {len(task_list)} tasks")
    elif tasks:
        task_list = tasks
        logger.info(f"📋 Custom tasks: {len(task_list)} tasks")
    else:
        # All tasks (warning: can take hours!)
        task_list = None
        logger.warning("⚠️  Running ALL MTEB tasks - this will take several hours!")

    # Run evaluation
    try:
        evaluation = MTEB(tasks=task_list)
        results = evaluation.run(
            model_wrapper,
            output_folder=f"results/{model_key}",
            eval_splits=["test"]
        )

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Evaluation complete for {model_key}")
        logger.info(f"{'='*60}\n")
        logger.info(f"Results saved to: results/{model_key}")

        # Calculate average score if available
        if results:
            scores = []
            for task_name, task_results in results.items():
                if "test" in task_results and "main_score" in task_results["test"]:
                    score = task_results["test"]["main_score"]
                    scores.append(score)
                    logger.info(f"  {task_name}: {score:.4f}")

            if scores:
                avg_score = np.mean(scores)
                logger.info(f"\n📊 Average Score: {avg_score:.4f}")

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


def compare_models(model_keys: List[str], tasks: List[str] = None):
    """
    Compare multiple models on the same tasks
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"🏆 Comparing models: {', '.join(model_keys)}")
    logger.info(f"{'='*60}\n")

    for model_key in model_keys:
        evaluate_model(model_key, tasks=tasks or QUICK_TASKS, quick=False)


def main():
    parser = argparse.ArgumentParser(description="MTEB Evaluation for Model2Vec models")
    parser.add_argument("--model", type=str, choices=list(MODELS.keys()),
                      help=f"Model to evaluate: {list(MODELS.keys())}")
    parser.add_argument("--all", action="store_true",
                      help="Compare all available models")
    parser.add_argument("--quick", action="store_true",
                      help="Run only quick lightweight tasks (~5 min)")
    parser.add_argument("--tasks", type=str, nargs="+",
                      help="Specific tasks to run (e.g., Banking77Classification)")
    parser.add_argument("--category", type=str, choices=list(TASK_CATEGORIES.keys()),
                      help=f"Task category: {list(TASK_CATEGORIES.keys())}")
    parser.add_argument("--reranking", action="store_true",
                      help="Evaluate reranking tasks only")

    args = parser.parse_args()

    # Determine tasks
    tasks = None
    if args.reranking:
        tasks = RERANKING_TASKS
        logger.info("🔄 Running reranking evaluation")
    elif args.category:
        tasks = TASK_CATEGORIES[args.category]
        logger.info(f"📁 Running {args.category} tasks")
    elif args.tasks:
        tasks = args.tasks

    # Run evaluation
    if args.all:
        compare_models(list(MODELS.keys()), tasks=tasks)
    elif args.model:
        evaluate_model(args.model, tasks=tasks, quick=args.quick)
    else:
        parser.print_help()
        print("\n💡 Examples:")
        print("  python scripts/evaluate_mteb.py --model gemma-768d --quick")
        print("  python scripts/evaluate_mteb.py --model qwen3-m2v --reranking")
        print("  python scripts/evaluate_mteb.py --all --category classification")


if __name__ == "__main__":
    main()
