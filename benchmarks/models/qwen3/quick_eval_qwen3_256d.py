#!/usr/bin/env python3
"""
Quick Quality Evaluation for Qwen3-256D Model2Vec

Evaluates the pre-distilled 256D Qwen3 model on:
- Semantic similarity
- Topic clustering
- Multilingual alignment

Direct comparison with Gemma-768D to determine best deployment choice.
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
    """Evaluate Qwen3-256D model quality"""

    logger.info("=" * 80)
    logger.info("🧪 Qwen3-256D Quality Evaluation")
    logger.info("=" * 80)

    try:
        from model2vec import StaticModel
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

    # Load model
    model_name = "Pringled/m2v-Qwen3-Embedding-0.6B"
    logger.info(f"\n📥 Loading model: {model_name}...")
    model = StaticModel.from_pretrained(model_name)
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
    logger.info(f"📊 OVERALL RESULTS - QWEN3-256D")
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

    # Load Gemma-768D results for comparison
    gemma_results_path = Path("gemma_768d_eval_results.json")
    if gemma_results_path.exists():
        with open(gemma_results_path, 'r') as f:
            gemma_results = json.load(f)

        logger.info(f"\n📈 HEAD-TO-HEAD COMPARISON:")
        logger.info(f"=" * 80)
        logger.info(f"\n   Metric                  | Qwen3-256D | Gemma-768D | Winner")
        logger.info(f"   " + "-" * 70)

        # Semantic
        qwen_sem = semantic_score
        gemma_sem = gemma_results['semantic_similarity']
        sem_winner = "Qwen3" if qwen_sem > gemma_sem else "Gemma" if gemma_sem > qwen_sem else "TIE"
        logger.info(f"   Semantic Similarity     | {qwen_sem:.4f}     | {gemma_sem:.4f}     | {sem_winner}")

        # Clustering
        qwen_clus = clustering_score
        gemma_clus = gemma_results['topic_clustering']
        clus_winner = "Qwen3" if qwen_clus > gemma_clus else "Gemma" if gemma_clus > qwen_clus else "TIE"
        logger.info(f"   Topic Clustering        | {qwen_clus:.4f}     | {gemma_clus:.4f}     | {clus_winner}")

        # Multilingual
        qwen_multi = multilingual_score
        gemma_multi = gemma_results['multilingual_alignment']
        multi_winner = "Qwen3" if qwen_multi > gemma_multi else "Gemma" if gemma_multi > qwen_multi else "TIE"
        logger.info(f"   Multilingual Alignment  | {qwen_multi:.4f}     | {gemma_multi:.4f}     | {multi_winner}")

        logger.info(f"   " + "-" * 70)

        # Overall
        qwen_overall = overall_quality
        gemma_overall = gemma_results['overall_quality']
        overall_winner = "Qwen3" if qwen_overall > gemma_overall else "Gemma" if gemma_overall > qwen_overall else "TIE"
        logger.info(f"   OVERALL QUALITY         | {qwen_overall:.4f}     | {gemma_overall:.4f}     | {overall_winner}")

        logger.info(f"\n   Dimensions:             | 256D       | 768D       |")
        logger.info(f"   Model Size:             | ~200MB     | ~400MB     |")

        logger.info(f"\n" + "=" * 80)
        if overall_winner == "Qwen3":
            logger.info(f"✅ WINNER: Qwen3-256D (smaller, faster, better quality)")
        elif overall_winner == "Gemma":
            logger.info(f"✅ WINNER: Gemma-768D (higher quality justifies larger size)")
        else:
            logger.info(f"⚖️  TIE: Choose based on size/speed vs dimensions tradeoff")
        logger.info(f"=" * 80)

    # Save results
    results = {
        "model": model_name,
        "dimensions": 256,
        "overall_quality": overall_quality,
        "semantic_similarity": semantic_score,
        "topic_clustering": clustering_score,
        "multilingual_alignment": multilingual_score,
        "silhouette_score": silhouette,
        "cluster_purity": purity,
        "avg_multilingual_similarity": multilingual_score,
        "assessment": assessment,
        "recommendation": recommendation,
    }

    results_path = Path("qwen3_256d_eval_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n💾 Results saved to {results_path}")

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
