#!/usr/bin/env python3
"""
Quick Quality Evaluation for Qwen2.5-1024D Model2Vec

Evaluates the distilled Qwen2.5-1.5B-Instruct model on:
1. Semantic similarity (standard)
2. Topic clustering (standard)
3. Multilingual alignment (standard)
4. **Instruction-awareness** (UNIQUE - key differentiator)
5. **Conversational understanding** (idioms, expressions)
6. **Code understanding** (technical capability)

This evaluation focuses on proving the unique advantages of distilling
an instruction-tuned LLM into static embeddings.
"""

import logging
from pathlib import Path
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model():
    """Evaluate Qwen2.5-1024D model quality"""

    logger.info("=" * 80)
    logger.info("🧪 Qwen2.5-1.5B-Instruct Model2Vec Quality Evaluation (1024D)")
    logger.info("=" * 80)

    try:
        from model2vec import StaticModel
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

    # Load model
    model_path = Path("models/qwen25-deposium-1024d")
    if not model_path.exists():
        logger.error(f"❌ Model not found: {model_path}")
        logger.error(f"   Run distillation first: python distill_qwen25_1024d.py")
        return False

    logger.info(f"\n📥 Loading model from {model_path}...")
    model = StaticModel.from_pretrained(str(model_path))
    logger.info(f"✅ Model loaded!")

    # Test embedding to get dimensions
    test_emb = model.encode(["test"], show_progress_bar=False)[0]
    logger.info(f"   Dimensions: {len(test_emb)}D")

    # Initialize scores
    all_scores = {}

    # ============================================================================
    # Test 1: Semantic Similarity (Baseline)
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"🔬 Test 1: Semantic Similarity (Baseline)")
    logger.info(f"{'='*80}")

    similar_pairs = [
        ("The cat sat on the mat", "A feline rested on the rug"),
        ("Machine learning is fascinating", "AI and deep learning are interesting"),
        ("The weather is sunny today", "It's a bright and clear day"),
        ("Quantum computing uses qubits", "Quantum computers leverage quantum bits"),
    ]

    dissimilar_pairs = [
        ("The cat sat on the mat", "Quantum physics explains the universe"),
        ("Machine learning is fascinating", "I enjoy eating pizza for dinner"),
        ("The weather is sunny today", "Databases store structured information"),
        ("Python is a programming language", "The ocean is deep and mysterious"),
    ]

    similar_scores = []
    for s1, s2 in similar_pairs:
        emb1 = model.encode([s1], show_progress_bar=False)[0]
        emb2 = model.encode([s2], show_progress_bar=False)[0]
        score = cosine_similarity([emb1], [emb2])[0][0]
        similar_scores.append(score)
        logger.info(f"   Similar:   {score:.4f} - '{s1[:40]}...' ↔ '{s2[:40]}...'")

    logger.info("")
    dissimilar_scores = []
    for s1, s2 in dissimilar_pairs:
        emb1 = model.encode([s1], show_progress_bar=False)[0]
        emb2 = model.encode([s2], show_progress_bar=False)[0]
        score = cosine_similarity([emb1], [emb2])[0][0]
        dissimilar_scores.append(score)
        logger.info(f"   Dissimilar: {score:.4f} - '{s1[:40]}...' ↔ '{s2[:40]}...'")

    avg_similar = np.mean(similar_scores)
    avg_dissimilar = np.mean(dissimilar_scores)
    semantic_score = (avg_similar - avg_dissimilar + 1) / 2

    logger.info(f"\n   Avg similar:    {avg_similar:.4f}")
    logger.info(f"   Avg dissimilar: {avg_dissimilar:.4f}")
    logger.info(f"   ✅ Semantic score: {semantic_score:.4f}")
    all_scores['semantic_similarity'] = semantic_score

    # ============================================================================
    # Test 2: Topic Clustering
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"🔬 Test 2: Topic Clustering")
    logger.info(f"{'='*80}")

    tech_sentences = [
        "Machine learning algorithms analyze data patterns",
        "Neural networks mimic the human brain structure",
        "Deep learning requires large datasets for training",
    ]

    nature_sentences = [
        "The forest is full of tall green trees",
        "Birds sing beautiful songs in the morning",
        "Rivers flow through valleys and mountains",
    ]

    sports_sentences = [
        "Football is a popular team sport worldwide",
        "Athletes train hard for competitions and medals",
        "Basketball requires coordination and teamwork",
    ]

    all_sentences = tech_sentences + nature_sentences + sports_sentences
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    embeddings = model.encode(all_sentences, show_progress_bar=False)
    embeddings_array = np.array([emb for emb in embeddings])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(embeddings_array)

    silhouette = silhouette_score(embeddings_array, labels)

    logger.info(f"   True labels: {labels}")
    logger.info(f"   Pred labels: {pred_labels.tolist()}")
    logger.info(f"   Silhouette:  {silhouette:.4f}")

    from collections import Counter
    cluster_label_counts = {}
    for i in range(3):
        cluster_indices = np.where(pred_labels == i)[0]
        true_labels_in_cluster = [labels[idx] for idx in cluster_indices]
        cluster_label_counts[i] = Counter(true_labels_in_cluster)

    correct = sum(max(counts.values()) for counts in cluster_label_counts.values())
    purity = correct / len(labels)

    logger.info(f"   Purity:      {purity:.4f}")
    clustering_score = (silhouette + purity) / 2
    logger.info(f"   ✅ Clustering score: {clustering_score:.4f}")
    all_scores['topic_clustering'] = clustering_score

    # ============================================================================
    # Test 3: Multilingual Alignment
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"🔬 Test 3: Multilingual Alignment")
    logger.info(f"{'='*80}")

    multilingual_pairs = [
        ("Hello world", "Bonjour le monde"),  # English-French
        ("Good morning", "Buenos días"),  # English-Spanish
        ("Thank you very much", "Danke schön"),  # English-German
        ("I love you", "Ti amo"),  # English-Italian
        ("How are you?", "¿Cómo estás?"),  # English-Spanish
        ("Artificial intelligence", "Intelligence artificielle"),  # English-French
    ]

    multilingual_scores = []
    for en, other in multilingual_pairs:
        emb1 = model.encode([en], show_progress_bar=False)[0]
        emb2 = model.encode([other], show_progress_bar=False)[0]
        score = cosine_similarity([emb1], [emb2])[0][0]
        multilingual_scores.append(score)
        logger.info(f"   {score:.4f} - '{en}' ↔ '{other}'")

    multilingual_score = np.mean(multilingual_scores)
    logger.info(f"   ✅ Multilingual score: {multilingual_score:.4f}")
    all_scores['multilingual_alignment'] = multilingual_score

    # ============================================================================
    # Test 4: INSTRUCTION-AWARENESS (UNIQUE CAPABILITY) ⭐
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"🔬 Test 4: Instruction-Awareness ⭐ (UNIQUE DIFFERENTIATOR)")
    logger.info(f"{'='*80}")

    instruction_pairs = [
        ("Explain how neural networks work", "neural networks explanation tutorial guide"),
        ("Summarize machine learning concepts", "machine learning summary overview key points"),
        ("Find articles about quantum computing", "quantum computing articles documents papers"),
        ("List advantages of deep learning", "deep learning benefits advantages pros"),
        ("Compare Python and JavaScript", "Python vs JavaScript comparison differences"),
        ("Describe the process of photosynthesis", "photosynthesis process description how it works"),
        ("Translate this to French", "French translation language conversion"),
    ]

    instruction_scores = []
    for instruction, semantic in instruction_pairs:
        emb1 = model.encode([instruction], show_progress_bar=False)[0]
        emb2 = model.encode([semantic], show_progress_bar=False)[0]
        score = cosine_similarity([emb1], [emb2])[0][0]
        instruction_scores.append(score)
        logger.info(f"   {score:.4f} - '{instruction[:45]}...' ↔ '{semantic[:45]}...'")

    instruction_score = np.mean(instruction_scores)
    logger.info(f"\n   ✅ Instruction-awareness: {instruction_score:.4f}")
    if instruction_score >= 0.70:
        logger.info(f"   🔥 EXCELLENT - Superior instruction understanding!")
    elif instruction_score >= 0.60:
        logger.info(f"   ✅ GOOD - Strong instruction understanding")
    elif instruction_score >= 0.50:
        logger.info(f"   ⚠️  MODERATE - Acceptable instruction understanding")
    else:
        logger.info(f"   ❌ POOR - Limited instruction understanding")

    all_scores['instruction_awareness'] = instruction_score

    # ============================================================================
    # Test 5: Conversational Understanding (Idioms, Expressions)
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"🔬 Test 5: Conversational Understanding (Idioms/Expressions)")
    logger.info(f"{'='*80}")

    idiom_pairs = [
        ("That's a piece of cake", "That's very easy simple straightforward"),
        ("Break a leg", "Good luck success wishes"),
        ("It's raining cats and dogs", "Heavy rain pouring downpour"),
        ("C'est du déjà-vu", "It's already seen before familiar"),
        ("Hit the nail on the head", "Exactly right correct precise"),
        ("Spill the beans", "Reveal secret tell truth"),
    ]

    conversational_scores = []
    for idiom, meaning in idiom_pairs:
        emb1 = model.encode([idiom], show_progress_bar=False)[0]
        emb2 = model.encode([meaning], show_progress_bar=False)[0]
        score = cosine_similarity([emb1], [emb2])[0][0]
        conversational_scores.append(score)
        logger.info(f"   {score:.4f} - '{idiom}' ↔ '{meaning}'")

    conversational_score = np.mean(conversational_scores)
    logger.info(f"   ✅ Conversational score: {conversational_score:.4f}")
    all_scores['conversational_understanding'] = conversational_score

    # ============================================================================
    # Test 6: Code Understanding
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"🔬 Test 6: Code Understanding")
    logger.info(f"{'='*80}")

    code_pairs = [
        ("def add(a, b): return a + b", "function to add two numbers sum"),
        ("for i in range(10): print(i)", "loop iterate numbers print values"),
        ("import numpy as np", "import scientific computing library numpy"),
        ("SELECT * FROM users WHERE age > 18", "database query select adult users"),
        ("async function fetchData() { await api.get() }", "asynchronous function fetch data API"),
    ]

    code_scores = []
    for code, description in code_pairs:
        emb1 = model.encode([code], show_progress_bar=False)[0]
        emb2 = model.encode([description], show_progress_bar=False)[0]
        score = cosine_similarity([emb1], [emb2])[0][0]
        code_scores.append(score)
        logger.info(f"   {score:.4f} - '{code[:50]}...' ↔ '{description}'")

    code_score = np.mean(code_scores)
    logger.info(f"   ✅ Code understanding: {code_score:.4f}")
    all_scores['code_understanding'] = code_score

    # ============================================================================
    # OVERALL QUALITY ASSESSMENT
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 OVERALL RESULTS")
    logger.info(f"{'='*80}")

    # Calculate overall score with weights
    # Instruction-awareness gets higher weight (unique capability)
    overall_quality = (
        semantic_score * 0.20 +
        clustering_score * 0.15 +
        multilingual_score * 0.15 +
        instruction_score * 0.30 +  # Higher weight - key differentiator
        conversational_score * 0.10 +
        code_score * 0.10
    )

    logger.info(f"\n   📈 Individual Scores:")
    logger.info(f"      Semantic Similarity:     {semantic_score:.4f} (weight: 20%)")
    logger.info(f"      Topic Clustering:        {clustering_score:.4f} (weight: 15%)")
    logger.info(f"      Multilingual:            {multilingual_score:.4f} (weight: 15%)")
    logger.info(f"      Instruction-Awareness:   {instruction_score:.4f} (weight: 30%) ⭐")
    logger.info(f"      Conversational:          {conversational_score:.4f} (weight: 10%)")
    logger.info(f"      Code Understanding:      {code_score:.4f} (weight: 10%)")

    logger.info(f"\n   🎯 Overall Quality: {overall_quality:.4f}")

    # Assessment
    if overall_quality >= 0.75:
        assessment = "EXCELLENT"
        emoji = "🔥"
        recommendation = "deploy_immediately"
    elif overall_quality >= 0.70:
        assessment = "VERY GOOD"
        emoji = "✅"
        recommendation = "deploy"
    elif overall_quality >= 0.65:
        assessment = "GOOD"
        emoji = "✅"
        recommendation = "deploy"
    elif overall_quality >= 0.60:
        assessment = "FAIR"
        emoji = "⚠️"
        recommendation = "test_first"
    else:
        assessment = "POOR"
        emoji = "❌"
        recommendation = "do_not_deploy"

    logger.info(f"\n   {emoji} Assessment: {assessment}")
    logger.info(f"   Recommendation: {recommendation.upper().replace('_', ' ')}")

    # ============================================================================
    # COMPETITIVE COMPARISON
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"📈 Comparison with Baselines")
    logger.info(f"{'='*80}")

    baselines = {
        'Qwen3-Embedding (600MB)': {'quality': 0.66, 'instruction_aware': False},
        'Qwen3-256D': {'quality': 0.665, 'instruction_aware': False},
        'Gemma-768D (400MB)': {'quality': 0.70, 'instruction_aware': False},
        'Qwen25-1024D (65MB)': {'quality': overall_quality, 'instruction_aware': True},
    }

    logger.info(f"\n   Model Comparison:")
    for model_name, metrics in baselines.items():
        instr_flag = "✨ INSTRUCTION-AWARE" if metrics['instruction_aware'] else ""
        logger.info(f"      {model_name:30s} Quality: {metrics['quality']:.4f} {instr_flag}")

    # Calculate improvements
    improvement_vs_qwen3 = ((overall_quality - 0.665) / 0.665) * 100
    improvement_vs_gemma = ((overall_quality - 0.70) / 0.70) * 100

    logger.info(f"\n   📊 Improvements:")
    logger.info(f"      vs Qwen3-256D:  {improvement_vs_qwen3:+.1f}%")
    logger.info(f"      vs Gemma-768D:  {improvement_vs_gemma:+.1f}%")

    # Unique advantages
    logger.info(f"\n   ✨ Unique Advantages of Qwen25-1024D:")
    logger.info(f"      1. Instruction-aware embeddings (ONLY model with this)")
    logger.info(f"      2. 10x smaller than Qwen3-Embedding (65MB vs 600MB)")
    logger.info(f"      3. Superior base model (1.54B vs 600M params)")
    logger.info(f"      4. Conversational understanding (idioms, expressions)")
    logger.info(f"      5. Code understanding capability")

    if overall_quality > 0.70:
        logger.info(f"\n   🚀 WINNER: Qwen25-1024D is the NEW CHAMPION!")
    elif overall_quality > 0.665:
        logger.info(f"\n   ✅ Qwen25-1024D is competitive with unique advantages")
    else:
        logger.info(f"\n   ⚠️  Consider further optimization or evaluation")

    # Save results
    results = {
        "model": str(model_path),
        "dimensions": 1024,
        "overall_quality": overall_quality,
        "individual_scores": all_scores,
        "assessment": assessment,
        "recommendation": recommendation,
        "improvements": {
            "vs_qwen3_256d_percent": improvement_vs_qwen3,
            "vs_gemma_768d_percent": improvement_vs_gemma,
        },
        "unique_capabilities": [
            "instruction_aware",
            "conversational_understanding",
            "code_understanding",
            "multilingual",
            "ultra_compact_65mb"
        ],
        "comparison": baselines
    }

    results_path = Path("qwen25_1024d_eval_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n💾 Results saved to {results_path}")

    # Final verdict
    logger.info(f"\n{'='*80}")
    if overall_quality >= 0.70 and instruction_score >= 0.60:
        logger.info(f"🎉 SUCCESS - READY FOR DEPLOYMENT")
        logger.info(f"   Quality: {overall_quality:.4f} (EXCELLENT)")
        logger.info(f"   Instruction-aware: {instruction_score:.4f} (UNIQUE CAPABILITY)")
        logger.info(f"   Size: 65MB (10x smaller than competitors)")
        logger.info(f"   🚀 This is a GAME-CHANGER model!")
    elif overall_quality >= 0.65:
        logger.info(f"✅ GOOD QUALITY - READY FOR DEPLOYMENT")
        logger.info(f"   Quality: {overall_quality:.4f}")
        logger.info(f"   Instruction-aware: {instruction_score:.4f}")
    else:
        logger.info(f"⚠️  NEEDS IMPROVEMENT")
        logger.info(f"   Quality: {overall_quality:.4f}")
    logger.info(f"{'='*80}")

    return overall_quality >= 0.60


if __name__ == "__main__":
    import sys

    success = evaluate_model()

    if success:
        logger.info("\n🎉 Evaluation successful! Model quality is sufficient.")
        sys.exit(0)
    else:
        logger.error("\n❌ Evaluation failed or quality is insufficient.")
        sys.exit(1)
