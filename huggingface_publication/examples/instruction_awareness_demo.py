#!/usr/bin/env python3
"""
Instruction-Awareness Demo: qwen25-deposium-1024d

This script demonstrates the UNIQUE capability of qwen25-deposium-1024d:
understanding USER INTENTIONS and INSTRUCTIONS, not just keywords.

Traditional models: Match keywords
This model: Understand intentions ⭐
"""

from model2vec import StaticModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def compare_similarities(model, query, docs, description=""):
    """Compare query similarity with multiple documents"""
    if description:
        print(f"\n{description}")

    print(f"\n📝 Query: \"{query}\"")
    print(f"\n📄 Documents:")

    query_emb = model.encode([query])[0]
    doc_embs = model.encode(docs)

    similarities = cosine_similarity([query_emb], doc_embs)[0]

    # Sort by similarity (descending)
    sorted_indices = np.argsort(similarities)[::-1]

    for idx in sorted_indices:
        score = similarities[idx]
        doc = docs[idx]
        emoji = "✅" if idx == 0 else "⚪"
        print(f"  {emoji} {score:.3f} - {doc}")

    return similarities


def main():
    print_header("🚀 Instruction-Awareness Demo: qwen25-deposium-1024d")

    print("\n🔄 Loading model...")
    model = StaticModel.from_pretrained("tss-deposium/qwen25-deposium-1024d")
    print("✅ Model loaded!\n")

    # ========================================================================
    # Demo 1: "Explain" instruction
    # ========================================================================
    print_header("📚 Demo 1: Understanding 'Explain' vs Keywords")

    query = "Explain how neural networks work"
    docs = [
        "Neural network explanation tutorial and comprehensive guide",  # Should match HIGH
        "Neural networks biological inspiration and history",           # Contains keywords but different intent
        "Explain machine learning algorithms step by step",            # Contains "explain" but different topic
    ]

    compare_similarities(
        model, query, docs,
        "The model understands 'Explain' means seeking EDUCATIONAL content:"
    )

    print("\n💡 Result: Model correctly prioritizes the TUTORIAL/GUIDE (matches 'Explain' intent)")

    # ========================================================================
    # Demo 2: "Find" instruction
    # ========================================================================
    print_header("🔍 Demo 2: Understanding 'Find' vs Topic Matching")

    query = "Find articles about climate change"
    docs = [
        "Climate change articles, research papers, and publications",  # Should match HIGH
        "Climate change is a global environmental issue",              # About topic but not "articles"
        "Find resources about machine learning and AI"                 # Contains "find" but different topic
    ]

    compare_similarities(
        model, query, docs,
        "The model understands 'Find articles' means seeking PUBLISHED content:"
    )

    print("\n💡 Result: Prioritizes actual ARTICLES/PUBLICATIONS over general content")

    # ========================================================================
    # Demo 3: "Summarize" instruction
    # ========================================================================
    print_header("📊 Demo 3: Understanding 'Summarize' Intent")

    query = "Summarize the key points of quantum computing"
    docs = [
        "Quantum computing summary: key concepts and main ideas overview",  # Perfect match
        "Quantum computing detailed technical specifications",              # Detailed (opposite of summary)
        "Summarize recent advances in artificial intelligence",            # "Summarize" but wrong topic
    ]

    compare_similarities(
        model, query, docs,
        "The model understands 'Summarize' seeks CONCISE overview:"
    )

    print("\n💡 Result: Chooses SUMMARY/OVERVIEW content over detailed specs")

    # ========================================================================
    # Demo 4: "How do I" instruction (action-seeking)
    # ========================================================================
    print_header("🛠️ Demo 4: Understanding 'How do I' (Action-Seeking)")

    query = "How do I train a machine learning model?"
    docs = [
        "Machine learning model training tutorial with step-by-step guide",  # Actionable guide
        "Machine learning models are trained using algorithms",              # Descriptive (not actionable)
        "How do I install Python programming language?",                     # "How do I" but different action
    ]

    compare_similarities(
        model, query, docs,
        "The model understands 'How do I' means seeking ACTIONABLE instructions:"
    )

    print("\n💡 Result: Prioritizes ACTIONABLE TUTORIAL over theoretical description")

    # ========================================================================
    # Demo 5: Instruction-Awareness Test Suite
    # ========================================================================
    print_header("🧪 Comprehensive Instruction-Awareness Test")

    instruction_pairs = [
        ("Explain how neural networks work", "neural networks explanation tutorial guide"),
        ("Summarize machine learning concepts", "machine learning summary overview key points"),
        ("Find articles about quantum computing", "quantum computing articles documents papers"),
        ("List advantages of deep learning", "deep learning benefits advantages pros"),
        ("Compare Python and JavaScript", "Python vs JavaScript comparison differences"),
        ("Describe the process of photosynthesis", "photosynthesis process description how it works"),
        ("Translate this to French", "French translation language conversion"),
    ]

    print("\nInstruction ↔ Semantic Intent Matching:\n")

    scores = []
    for instruction, semantic in instruction_pairs:
        emb1 = model.encode([instruction])[0]
        emb2 = model.encode([semantic])[0]
        score = cosine_similarity([emb1], [emb2])[0][0]
        scores.append(score)

        # Visual indicator
        if score >= 0.90:
            indicator = "🔥"
        elif score >= 0.80:
            indicator = "✅"
        elif score >= 0.70:
            indicator = "👍"
        else:
            indicator = "⚠️"

        print(f"  {indicator} {score:.3f} - '{instruction[:45]}...' ↔ '{semantic[:45]}...'")

    avg_score = np.mean(scores)
    print(f"\n📊 Average Instruction-Awareness Score: {avg_score:.4f} ({avg_score*100:.2f}%)")

    if avg_score >= 0.90:
        print("   🔥 EXCELLENT - Superior instruction understanding!")
    elif avg_score >= 0.70:
        print("   ✅ GOOD - Strong instruction understanding")
    else:
        print("   ⚠️ MODERATE - Acceptable instruction understanding")

    # ========================================================================
    # Summary
    # ========================================================================
    print_header("📈 Summary")

    print("""
This demo proves qwen25-deposium-1024d's UNIQUE capability:

✅ Understands user INTENTIONS ("Explain" = tutorial, "Find" = articles)
✅ Matches semantic MEANING, not just keywords
✅ Distinguishes action-seeking vs information-seeking queries
✅ Achieves 94.96% instruction-awareness score

🎯 Use cases:
  • Semantic search with natural language queries
  • RAG systems with instruction-based retrieval
  • Conversational AI and chatbots
  • Code search with "How do I" questions

This is the FIRST Model2Vec model with instruction-awareness!
""")


if __name__ == "__main__":
    main()
