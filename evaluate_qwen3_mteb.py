#!/usr/bin/env python3
"""
MTEB Evaluation for Qwen3 Model2Vec

Evaluates Pringled/m2v-Qwen3-Embedding-0.6B on MTEB tasks.
Target: >0.65 average score (vs gemma baseline 0.788)
"""

import logging
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_qwen3_mteb():
    """Run MTEB evaluation on Qwen3 Model2Vec"""

    logger.info("=" * 80)
    logger.info("📊 MTEB Evaluation: Qwen3 Model2Vec")
    logger.info("=" * 80)

    try:
        import mteb
        from model2vec import StaticModel
        logger.info("✅ Dependencies imported")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Install: pip install mteb model2vec")
        return False

    model_name = "Pringled/m2v-Qwen3-Embedding-0.6B"

    logger.info(f"\n📥 Loading model: {model_name}")
    try:
        model = StaticModel.from_pretrained(model_name)
        logger.info("✅ Model loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

    # Create a wrapper for MTEB compatibility
    class Model2VecWrapper:
        """Wrapper to make Model2Vec compatible with MTEB"""

        def __init__(self, model):
            self.model = model

        def encode(self, sentences, **kwargs):
            """Encode sentences to embeddings"""
            return self.model.encode(sentences)

    wrapped_model = Model2VecWrapper(model)

    # Select MTEB tasks (subset for faster evaluation)
    # Full evaluation takes hours, so we test key categories
    logger.info("\n📋 MTEB Tasks (subset for speed):")
    logger.info("  • Retrieval: ArguAna, NFCorpus")
    logger.info("  • Clustering: ArxivClusteringP2P")
    logger.info("  • Classification: AmazonReviewsClassification")
    logger.info("  • STS: STS22 (multilingual)")
    logger.info("  • Reranking: AskUbuntuDupQuestions")

    tasks = [
        "ArguAna",
        "NFCorpus",
        "ArxivClusteringP2P",
        "AmazonReviewsClassification",
        "STS22",
        "AskUbuntuDupQuestions",
    ]

    logger.info(f"\n🚀 Starting evaluation on {len(tasks)} tasks...")
    logger.info("This will take 10-30 minutes depending on CPU...")

    results = {}

    for task_name in tasks:
        try:
            logger.info(f"\n▶️  Evaluating: {task_name}")

            # Load task
            task = mteb.get_task(task_name)

            # Run evaluation
            task_results = task.evaluate(wrapped_model)

            # Extract main score
            if hasattr(task_results, 'scores'):
                scores = task_results.scores
                if isinstance(scores, dict):
                    main_score = scores.get('test', scores.get('validation', scores.get('dev', None)))
                    if isinstance(main_score, dict):
                        # For retrieval tasks, use ndcg@10
                        main_score = main_score.get('ndcg_at_10', main_score.get('map', main_score.get('accuracy', None)))
                else:
                    main_score = scores
            else:
                main_score = None

            results[task_name] = {
                'score': main_score,
                'full_results': task_results
            }

            logger.info(f"✅ {task_name}: {main_score:.4f}" if main_score else f"✅ {task_name}: completed")

        except Exception as e:
            logger.error(f"❌ Failed on {task_name}: {e}")
            results[task_name] = {'error': str(e)}

    # Calculate average score
    valid_scores = [r['score'] for r in results.values() if 'score' in r and r['score'] is not None]

    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
    else:
        avg_score = None

    logger.info("\n" + "=" * 80)
    logger.info("📊 MTEB EVALUATION RESULTS")
    logger.info("=" * 80)

    logger.info(f"\n📈 Task Scores:")
    for task_name, result in results.items():
        if 'error' in result:
            logger.info(f"  ❌ {task_name}: {result['error']}")
        elif result['score'] is not None:
            logger.info(f"  ✅ {task_name}: {result['score']:.4f}")
        else:
            logger.info(f"  ⚠️  {task_name}: No score extracted")

    if avg_score is not None:
        logger.info(f"\n🎯 Average Score: {avg_score:.4f}")
        logger.info(f"   Target: >0.65")
        logger.info(f"   Gemma baseline: 0.788")
        logger.info(f"   Quality loss: {(0.788 - avg_score) / 0.788 * 100:.1f}%")

        if avg_score >= 0.65:
            logger.info("\n✅ QUALITY ACCEPTABLE FOR DEPLOYMENT!")
            logger.info("🚂 Ready for Railway deployment")
        else:
            logger.info(f"\n⚠️  QUALITY BELOW TARGET ({avg_score:.4f} < 0.65)")
            logger.info("📋 Options:")
            logger.info("  1. Deploy anyway (710x speedup might justify quality loss)")
            logger.info("  2. Try Option B: Custom distill Qwen3 with 596k corpus")
            logger.info("  3. Look for higher-quality pre-distilled models")

    # Save results
    output_file = Path("qwen3_mteb_results.json")
    with open(output_file, 'w') as f:
        # Convert results to serializable format
        serializable_results = {}
        for task_name, result in results.items():
            if 'score' in result:
                serializable_results[task_name] = {'score': float(result['score']) if result['score'] is not None else None}
            else:
                serializable_results[task_name] = result

        json.dump({
            'model': model_name,
            'average_score': float(avg_score) if avg_score is not None else None,
            'task_results': serializable_results
        }, f, indent=2)

    logger.info(f"\n💾 Results saved to: {output_file}")

    logger.info("\n" + "=" * 80)

    return avg_score is not None and avg_score >= 0.65


if __name__ == "__main__":
    import sys

    success = evaluate_qwen3_mteb()

    if success:
        logger.info("\n🎉 Evaluation successful - quality acceptable!")
        sys.exit(0)
    else:
        logger.info("\n⚠️  Evaluation complete - review results")
        sys.exit(0)  # Exit 0 even if below target, let user decide
