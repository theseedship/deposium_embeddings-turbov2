#!/usr/bin/env python3
"""
Quick Quality Evaluation for Qwen3 1024D Model2Vec

Evaluates the custom distilled 1024D model on:
- Semantic similarity
- Topic clustering
- Multilingual alignment
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
    """Evaluate Qwen3 1024D model quality"""

    logger.info("=" * 80)
    logger.info("🧪 Qwen3 1024D Quality Evaluation")
    logger.info("=" * 80)

    try:
        from model2vec import StaticModel
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

    # Load model
    model_path = Path("models/qwen3-deposium-1024d")
    if not model_path.exists():
        logger.error(f"❌ Model not found: {model_path}")
        return False

    logger.info(f"\n📥 Loading model from {model_path}...")
    model = StaticModel.from_pretrained(str(model_path))
    logger.info(f"✅ Model loaded!")
    # Test embedding to get dimensions
    test_emb = model.encode(["test"], show_progress_bar=False)[0]
    logger.info(f"   Dimensions: {len(test_emb)}D")

    # 1. Semantic Similarity Test
    logger.info(f"\n🔬 Test 1: Semantic Similarity")
    similar_pairs = [
        ("The cat sat on the mat", "A feline rested on the rug"),
        ("Machine learning is fascinating", "AI and deep learning are interesting"),
        ("The weather is sunny today", "It's a bright and clear day"),
    ]

    dissimilar_pairs = [
        ("The cat sat on the mat", "Quantum physics explains the universe"),
        ("Machine learning is fascinating", "I enjoy eating pizza"),
        ("The weather is sunny today", "Databases store information"),
    ]

    similar_scores = []
    for s1, s2 in similar_pairs:
        emb1 = model.encode([s1], show_progress_bar=False)[0]
        emb2 = model.encode([s2], show_progress_bar=False)[0]
        score = cosine_similarity([emb1], [emb2])[0][0]
        similar_scores.append(score)
        logger.info(f"   Similar: {score:.4f} - '{s1[:30]}...' <-> '{s2[:30]}...'")

    dissimilar_scores = []
    for s1, s2 in dissimilar_pairs:
        emb1 = model.encode([s1], show_progress_bar=False)[0]
        emb2 = model.encode([s2], show_progress_bar=False)[0]
        score = cosine_similarity([emb1], [emb2])[0][0]
        dissimilar_scores.append(score)
        logger.info(f"   Dissimilar: {score:.4f} - '{s1[:30]}...' <-> '{s2[:30]}...'")

    avg_similar = np.mean(similar_scores)
    avg_dissimilar = np.mean(dissimilar_scores)
    semantic_score = (avg_similar - avg_dissimilar + 1) / 2  # Normalize to 0-1

    logger.info(f"\n   Avg similar: {avg_similar:.4f}")
    logger.info(f"   Avg dissimilar: {avg_dissimilar:.4f}")
    logger.info(f"   ✅ Semantic score: {semantic_score:.4f}")

    # 2. Topic Clustering Test
    logger.info(f"\n🔬 Test 2: Topic Clustering")
    tech_sentences = [
        "Machine learning algorithms analyze data patterns",
        "Neural networks mimic the human brain",
        "Deep learning requires large datasets",
    ]

    nature_sentences = [
        "The forest is full of tall trees",
        "Birds sing in the morning sunlight",
        "Rivers flow through green valleys",
    ]

    sports_sentences = [
        "Football is a popular team sport",
        "Athletes train hard for competitions",
        "Basketball requires good coordination",
    ]

    all_sentences = tech_sentences + nature_sentences + sports_sentences
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    embeddings = model.encode(all_sentences, show_progress_bar=False)
    embeddings_array = np.array([emb for emb in embeddings])

    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(embeddings_array)

    # Silhouette score (measures cluster quality)
    silhouette = silhouette_score(embeddings_array, labels)

    logger.info(f"   True labels: {labels}")
    logger.info(f"   Pred labels: {pred_labels.tolist()}")
    logger.info(f"   ✅ Silhouette score: {silhouette:.4f}")

    # Calculate purity
    from collections import Counter
    cluster_label_counts = {}
    for i in range(3):
        cluster_indices = np.where(pred_labels == i)[0]
        true_labels_in_cluster = [labels[idx] for idx in cluster_indices]
        cluster_label_counts[i] = Counter(true_labels_in_cluster)

    correct = sum(max(counts.values()) for counts in cluster_label_counts.values())
    purity = correct / len(labels)

    logger.info(f"   Cluster purity: {purity:.4f}")
    clustering_score = (silhouette + purity) / 2
    logger.info(f"   ✅ Clustering score: {clustering_score:.4f}")

    # 3. Multilingual Alignment Test
    logger.info(f"\n🔬 Test 3: Multilingual Alignment")
    multilingual_pairs = [
        ("Hello world", "Bonjour le monde"),  # English-French
        ("Good morning", "Buenos días"),  # English-Spanish
        ("Thank you", "Danke schön"),  # English-German
        ("I love you", "Ti amo"),  # English-Italian
    ]

    multilingual_scores = []
    for en, other in multilingual_pairs:
        emb1 = model.encode([en], show_progress_bar=False)[0]
        emb2 = model.encode([other], show_progress_bar=False)[0]
        score = cosine_similarity([emb1], [emb2])[0][0]
        multilingual_scores.append(score)
        logger.info(f"   {score:.4f} - '{en}' <-> '{other}'")

    multilingual_score = np.mean(multilingual_scores)
    logger.info(f"   ✅ Multilingual score: {multilingual_score:.4f}")

    # 4. Overall Quality Score
    logger.info(f"\n" + "=" * 80)
    logger.info(f"📊 OVERALL RESULTS")
    logger.info(f"=" * 80)

    overall_quality = (semantic_score + clustering_score + multilingual_score) / 3

    logger.info(f"\n   Semantic Similarity: {semantic_score:.4f}")
    logger.info(f"   Topic Clustering:    {clustering_score:.4f}")
    logger.info(f"   Multilingual:        {multilingual_score:.4f}")
    logger.info(f"\n   🎯 Overall Quality:  {overall_quality:.4f}")

    # Quality assessment
    if overall_quality >= 0.70:
        assessment = "EXCELLENT"
        emoji = "🔥"
        recommendation = "deploy"
    elif overall_quality >= 0.60:
        assessment = "GOOD"
        emoji = "✅"
        recommendation = "deploy"
    elif overall_quality >= 0.50:
        assessment = "FAIR"
        emoji = "⚠️"
        recommendation = "test_first"
    else:
        assessment = "POOR"
        emoji = "❌"
        recommendation = "do_not_deploy"

    logger.info(f"\n   {emoji} Assessment: {assessment}")
    logger.info(f"   Recommendation: {recommendation.upper()}")

    # Comparison with 256D
    logger.info(f"\n📈 Comparison with 256D:")
    logger.info(f"   256D Quality: 0.6651 (GOOD)")
    logger.info(f"   1024D Quality: {overall_quality:.4f} ({assessment})")
    improvement = ((overall_quality - 0.6651) / 0.6651) * 100
    logger.info(f"   Improvement: {improvement:+.1f}%")

    # Save results
    results = {
        "model": str(model_path),
        "dimensions": 1024,
        "overall_quality": overall_quality,
        "semantic_similarity": semantic_score,
        "topic_clustering": clustering_score,
        "multilingual_alignment": multilingual_score,
        "silhouette_score": silhouette,
        "cluster_purity": purity,
        "avg_multilingual_similarity": multilingual_score,
        "assessment": assessment,
        "recommendation": recommendation,
        "comparison_256d": {
            "quality_256d": 0.6651,
            "quality_1024d": overall_quality,
            "improvement_percent": improvement
        }
    }

    results_path = Path("qwen3_1024d_eval_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n💾 Results saved to {results_path}")

    logger.info(f"\n" + "=" * 80)
    if recommendation == "deploy":
        logger.info(f"✅ READY FOR DEPLOYMENT")
    else:
        logger.info(f"⚠️  NEEDS MORE TESTING")
    logger.info(f"=" * 80)

    return overall_quality >= 0.60


if __name__ == "__main__":
    import sys

    success = evaluate_model()

    if success:
        logger.info("\n🎉 Evaluation successful! Model quality is good.")
        sys.exit(0)
    else:
        logger.error("\n❌ Evaluation failed or quality is insufficient.")
        sys.exit(1)
