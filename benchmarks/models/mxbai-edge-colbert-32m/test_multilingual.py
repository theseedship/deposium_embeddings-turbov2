#!/usr/bin/env python3
"""
Test Multilingue: mxbai-edge-colbert-v0-32m
Vérifie si le modèle comprend le français, l'allemand, l'espagnol
"""

import logging
from pylate import models
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def maxsim_score(query_emb, doc_emb):
    """Calculate MaxSim score"""
    if not isinstance(query_emb, torch.Tensor):
        query_emb = torch.from_numpy(query_emb)
    if not isinstance(doc_emb, torch.Tensor):
        doc_emb = torch.from_numpy(doc_emb)

    query_emb = query_emb / query_emb.norm(dim=1, keepdim=True)
    doc_emb = doc_emb / doc_emb.norm(dim=1, keepdim=True)

    sim_matrix = torch.matmul(query_emb, doc_emb.T)
    max_sims, _ = torch.max(sim_matrix, dim=1)
    score = torch.mean(max_sims).item()

    return score

def main():
    logger.info("🌍 Test Multilingue - mxbai-edge-colbert-v0-32m")
    logger.info("=" * 80)

    # Load model
    logger.info("📥 Loading model...")
    model = models.ColBERT(
        model_name_or_path="mixedbread-ai/mxbai-edge-colbert-v0-32m",
    )
    logger.info("✅ Model loaded\n")

    # Test pairs: similar meaning in different languages
    test_pairs = [
        # French
        ("Le chat est assis sur le tapis", "Un félin repose sur le tapis", "🇫🇷 French"),
        ("L'apprentissage automatique est fascinant", "L'IA et le deep learning sont intéressants", "🇫🇷 French"),

        # German
        ("Die Katze sitzt auf der Matte", "Eine Katze ruht auf dem Teppich", "🇩🇪 German"),
        ("Maschinelles Lernen ist faszinierend", "KI und Deep Learning sind interessant", "🇩🇪 German"),

        # Spanish
        ("El gato está sentado en la alfombra", "Un felino descansa en la alfombra", "🇪🇸 Spanish"),
        ("El aprendizaje automático es fascinante", "La IA y el aprendizaje profundo son interesantes", "🇪🇸 Spanish"),

        # English (baseline)
        ("The cat sat on the mat", "A feline rested on the rug", "🇬🇧 English"),
        ("Machine learning is fascinating", "AI and deep learning are interesting", "🇬🇧 English"),
    ]

    logger.info("📊 Testing similar pairs across languages:\n")

    results = {}
    for s1, s2, lang in test_pairs:
        queries_emb = model.encode([s1], is_query=True)
        docs_emb = model.encode([s2], is_query=False)

        score = maxsim_score(queries_emb[0], docs_emb[0])

        lang_key = lang.split()[1]  # Extract language name
        if lang_key not in results:
            results[lang_key] = []
        results[lang_key].append(score)

        logger.info(f"{lang}: {score:.4f}")
        logger.info(f"   '{s1}' ↔ '{s2}'\n")

    logger.info("=" * 80)
    logger.info("📊 SUMMARY BY LANGUAGE\n")

    for lang, scores in results.items():
        avg_score = sum(scores) / len(scores)
        logger.info(f"{lang}: {avg_score:.4f} (avg of {len(scores)} pairs)")

    # Conclusion
    english_avg = sum(results["English"]) / len(results["English"])
    logger.info(f"\n{'='*80}")
    logger.info("🎯 CONCLUSION\n")

    logger.info(f"English baseline: {english_avg:.4f}")

    for lang, scores in results.items():
        if lang == "English":
            continue
        avg = sum(scores) / len(scores)
        diff = avg - english_avg
        pct_diff = (diff / english_avg) * 100

        if pct_diff > -10:  # Less than 10% degradation
            verdict = "✅ Good multilingual support"
        elif pct_diff > -20:
            verdict = "⚠️ Moderate support"
        else:
            verdict = "❌ Poor multilingual support"

        logger.info(f"{lang}: {avg:.4f} ({pct_diff:+.1f}% vs English) - {verdict}")

    logger.info("\n✅ Test complete!")

if __name__ == "__main__":
    main()
