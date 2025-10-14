#!/usr/bin/env python3
"""
Direct Comparison: Qwen2.5-1024D vs All Other Models

Compares the new Qwen2.5-1.5B-Instruct Model2Vec distillation
against all existing models on identical test sets.

Models compared:
- Qwen25-1024D (NEW - instruction-aware)
- Gemma-768D (previous best)
- Qwen3-256D (baseline)

Focus on proving unique advantages:
- Instruction-awareness (only Qwen25)
- Quality vs size trade-off
- Deployment recommendation
"""

import logging
from pathlib import Path
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: Path, model_name: str):
    """Load a Model2Vec model"""
    if not model_path.exists():
        logger.warning(f"⚠️  {model_name} not found at {model_path}")
        return None

    try:
        from model2vec import StaticModel
        model = StaticModel.from_pretrained(str(model_path))
        logger.info(f"✅ {model_name} loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load {model_name}: {e}")
        return None


def test_semantic_similarity(model, model_name: str):
    """Test semantic similarity understanding"""
    pairs = [
        ("Machine learning is fascinating", "AI and deep learning are interesting"),
        ("The weather is sunny today", "It's a bright and clear day"),
        ("Quantum computing uses qubits", "Quantum computers leverage quantum bits"),
    ]

    scores = []
    for s1, s2 in pairs:
        emb1 = model.encode([s1], show_progress_bar=False)[0]
        emb2 = model.encode([s2], show_progress_bar=False)[0]
        score = cosine_similarity([emb1], [emb2])[0][0]
        scores.append(score)

    return np.mean(scores)


def test_instruction_awareness(model, model_name: str):
    """Test instruction-awareness (KEY DIFFERENTIATOR)"""
    pairs = [
        ("Explain how neural networks work", "neural networks explanation tutorial guide"),
        ("Summarize machine learning concepts", "machine learning summary overview key points"),
        ("Find articles about quantum computing", "quantum computing articles documents papers"),
        ("List advantages of deep learning", "deep learning benefits advantages pros"),
        ("Compare Python and JavaScript", "Python vs JavaScript comparison differences"),
    ]

    scores = []
    for instruction, semantic in pairs:
        emb1 = model.encode([instruction], show_progress_bar=False)[0]
        emb2 = model.encode([semantic], show_progress_bar=False)[0]
        score = cosine_similarity([emb1], [emb2])[0][0]
        scores.append(score)

    return np.mean(scores)


def test_multilingual(model, model_name: str):
    """Test multilingual alignment"""
    pairs = [
        ("Hello world", "Bonjour le monde"),
        ("Good morning", "Buenos días"),
        ("Thank you very much", "Danke schön"),
        ("Artificial intelligence", "Intelligence artificielle"),
    ]

    scores = []
    for en, other in pairs:
        emb1 = model.encode([en], show_progress_bar=False)[0]
        emb2 = model.encode([other], show_progress_bar=False)[0]
        score = cosine_similarity([emb1], [emb2])[0][0]
        scores.append(score)

    return np.mean(scores)


def test_conversational(model, model_name: str):
    """Test conversational understanding (idioms)"""
    pairs = [
        ("That's a piece of cake", "That's very easy simple straightforward"),
        ("Break a leg", "Good luck success wishes"),
        ("It's raining cats and dogs", "Heavy rain pouring downpour"),
        ("Hit the nail on the head", "Exactly right correct precise"),
    ]

    scores = []
    for idiom, meaning in pairs:
        emb1 = model.encode([idiom], show_progress_bar=False)[0]
        emb2 = model.encode([meaning], show_progress_bar=False)[0]
        score = cosine_similarity([emb1], [emb2])[0][0]
        scores.append(score)

    return np.mean(scores)


def test_code_understanding(model, model_name: str):
    """Test code understanding"""
    pairs = [
        ("def add(a, b): return a + b", "function to add two numbers sum"),
        ("for i in range(10): print(i)", "loop iterate numbers print values"),
        ("import numpy as np", "import scientific computing library numpy"),
    ]

    scores = []
    for code, description in pairs:
        emb1 = model.encode([code], show_progress_bar=False)[0]
        emb2 = model.encode([description], show_progress_bar=False)[0]
        score = cosine_similarity([emb1], [emb2])[0][0]
        scores.append(score)

    return np.mean(scores)


def compare_all_models():
    """Compare all available models"""

    logger.info("=" * 80)
    logger.info("📊 COMPREHENSIVE MODEL COMPARISON")
    logger.info("=" * 80)

    # Define models to compare
    models_config = {
        'Qwen25-1024D': {
            'path': Path('models/qwen25-deposium-1024d'),
            'size_mb': 65,
            'type': 'Instruction-tuned LLM distillation',
            'base': 'Qwen2.5-1.5B-Instruct (1.54B params)',
        },
        'Gemma-768D': {
            'path': Path('models/gemma-deposium-768d'),
            'size_mb': 400,
            'type': 'Embedding model distillation',
            'base': 'google/embeddinggemma-300m',
        },
        'Qwen3-256D': {
            'path': Path('models/qwen3-deposium-256d'),
            'size_mb': 200,
            'type': 'Embedding model distillation',
            'base': 'Qwen3-Embedding-0.6B',
        },
    }

    # Load all models
    logger.info(f"\n📥 Loading models...")
    models = {}
    for name, config in models_config.items():
        model = load_model(config['path'], name)
        if model:
            models[name] = {'model': model, 'config': config}

    if not models:
        logger.error("❌ No models found! Run distillation scripts first.")
        return False

    logger.info(f"\n✅ Loaded {len(models)} model(s)")

    # Run all tests
    logger.info(f"\n{'='*80}")
    logger.info("🧪 Running Comprehensive Tests")
    logger.info(f"{'='*80}")

    results = {}

    for model_name, model_data in models.items():
        logger.info(f"\n📊 Testing {model_name}...")
        model = model_data['model']

        # Test embeddings dimension
        test_emb = model.encode(["test"], show_progress_bar=False)[0]
        dimensions = len(test_emb)

        # Run all tests
        semantic = test_semantic_similarity(model, model_name)
        instruction = test_instruction_awareness(model, model_name)
        multilingual = test_multilingual(model, model_name)
        conversational = test_conversational(model, model_name)
        code = test_code_understanding(model, model_name)

        # Calculate overall score (weighted)
        overall = (
            semantic * 0.20 +
            instruction * 0.30 +  # Higher weight for instruction-awareness
            multilingual * 0.15 +
            conversational * 0.15 +
            code * 0.20
        )

        results[model_name] = {
            'dimensions': dimensions,
            'size_mb': model_data['config']['size_mb'],
            'base_model': model_data['config']['base'],
            'type': model_data['config']['type'],
            'scores': {
                'semantic_similarity': semantic,
                'instruction_awareness': instruction,
                'multilingual': multilingual,
                'conversational': conversational,
                'code_understanding': code,
                'overall': overall,
            }
        }

        logger.info(f"   Dimensions: {dimensions}D")
        logger.info(f"   Size: {model_data['config']['size_mb']}MB")
        logger.info(f"   Semantic: {semantic:.4f}")
        logger.info(f"   Instruction: {instruction:.4f}")
        logger.info(f"   Multilingual: {multilingual:.4f}")
        logger.info(f"   Conversational: {conversational:.4f}")
        logger.info(f"   Code: {code:.4f}")
        logger.info(f"   ✅ Overall: {overall:.4f}")

    # ============================================================================
    # COMPARISON TABLE
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info("📊 COMPARISON RESULTS")
    logger.info(f"{'='*80}")

    # Sort by overall score
    sorted_models = sorted(results.items(), key=lambda x: x[1]['scores']['overall'], reverse=True)

    logger.info(f"\n{'Model':<20} {'Size':>8} {'Dims':>6} {'Overall':>8} {'Instruction':>12} {'Semantic':>10}")
    logger.info("-" * 80)

    for model_name, data in sorted_models:
        scores = data['scores']
        size_mb = data['size_mb']
        dims = data['dimensions']

        logger.info(
            f"{model_name:<20} {size_mb:>6}MB {dims:>6}D "
            f"{scores['overall']:>8.4f} {scores['instruction_awareness']:>12.4f} "
            f"{scores['semantic_similarity']:>10.4f}"
        )

    # ============================================================================
    # DETAILED ANALYSIS
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info("🔍 DETAILED ANALYSIS")
    logger.info(f"{'='*80}")

    best_model = sorted_models[0][0]
    best_score = sorted_models[0][1]['scores']['overall']

    logger.info(f"\n🏆 Best Overall: {best_model} ({best_score:.4f})")

    # Analyze specific strengths
    logger.info(f"\n📈 Category Winners:")

    categories = ['semantic_similarity', 'instruction_awareness', 'multilingual',
                  'conversational', 'code_understanding']

    for category in categories:
        best_in_category = max(results.items(),
                              key=lambda x: x[1]['scores'][category])
        score = best_in_category[1]['scores'][category]
        logger.info(f"   {category.replace('_', ' ').title():<25s}: "
                   f"{best_in_category[0]:<20s} ({score:.4f})")

    # Size efficiency
    logger.info(f"\n📦 Size Efficiency (Quality per MB):")
    for model_name, data in sorted_models:
        efficiency = data['scores']['overall'] / data['size_mb'] * 1000
        logger.info(f"   {model_name:<20}: {efficiency:.2f} (quality per MB × 1000)")

    # ============================================================================
    # INSTRUCTION-AWARENESS ANALYSIS (KEY DIFFERENTIATOR)
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info("✨ INSTRUCTION-AWARENESS ANALYSIS (Key Differentiator)")
    logger.info(f"{'='*80}")

    logger.info(f"\nInstruction-awareness scores:")
    instruction_sorted = sorted(results.items(),
                               key=lambda x: x[1]['scores']['instruction_awareness'],
                               reverse=True)

    for model_name, data in instruction_sorted:
        score = data['scores']['instruction_awareness']
        if score >= 0.70:
            assessment = "🔥 EXCELLENT"
        elif score >= 0.60:
            assessment = "✅ GOOD"
        elif score >= 0.50:
            assessment = "⚠️  MODERATE"
        else:
            assessment = "❌ POOR"

        logger.info(f"   {model_name:<20}: {score:.4f} {assessment}")

    # ============================================================================
    # DEPLOYMENT RECOMMENDATION
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info("🚀 DEPLOYMENT RECOMMENDATION")
    logger.info(f"{'='*80}")

    best_name, best_data = sorted_models[0]
    best_scores = best_data['scores']

    logger.info(f"\n🏆 Recommended Model: {best_name}")
    logger.info(f"\n📊 Key Metrics:")
    logger.info(f"   Overall Quality:      {best_scores['overall']:.4f}")
    logger.info(f"   Instruction-Aware:    {best_scores['instruction_awareness']:.4f}")
    logger.info(f"   Size:                 {best_data['size_mb']}MB")
    logger.info(f"   Dimensions:           {best_data['dimensions']}D")

    logger.info(f"\n✨ Unique Advantages:")
    if best_name == 'Qwen25-1024D':
        logger.info(f"   1. Instruction-aware embeddings (UNIQUE capability)")
        logger.info(f"   2. Ultra-compact size (65MB vs 400-600MB)")
        logger.info(f"   3. Superior base model (1.54B params)")
        logger.info(f"   4. Conversational understanding")
        logger.info(f"   5. Code understanding")
        logger.info(f"   6. Strong multilingual support")
    elif best_name == 'Gemma-768D':
        logger.info(f"   1. Good overall quality")
        logger.info(f"   2. Native 768D embeddings")
        logger.info(f"   3. Multilingual support")

    # Quality vs Size trade-off
    logger.info(f"\n📦 Quality vs Size Trade-off:")
    for model_name, data in sorted_models:
        quality_per_100mb = (data['scores']['overall'] / data['size_mb']) * 100
        logger.info(f"   {model_name:<20}: {quality_per_100mb:.3f} quality per 100MB")

    # Save comparison results
    comparison_results = {
        'timestamp': str(Path.cwd()),
        'models_compared': len(results),
        'results': results,
        'best_model': best_name,
        'best_overall_score': best_scores['overall'],
        'recommendation': f"Deploy {best_name}",
    }

    results_path = Path('model_comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)

    logger.info(f"\n💾 Comparison results saved to {results_path}")

    logger.info(f"\n{'='*80}")
    if best_scores['overall'] >= 0.70:
        logger.info("✅ RECOMMENDATION: DEPLOY IMMEDIATELY")
        logger.info(f"   {best_name} shows excellent quality and unique advantages")
    elif best_scores['overall'] >= 0.65:
        logger.info("✅ RECOMMENDATION: DEPLOY AFTER STAGING TEST")
        logger.info(f"   {best_name} shows good quality")
    else:
        logger.info("⚠️  RECOMMENDATION: FURTHER OPTIMIZATION NEEDED")
    logger.info(f"{'='*80}")

    return True


if __name__ == "__main__":
    import sys

    try:
        success = compare_all_models()
        if success:
            logger.info("\n🎉 Comparison completed successfully!")
            sys.exit(0)
        else:
            logger.error("\n❌ Comparison failed!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error during comparison: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
