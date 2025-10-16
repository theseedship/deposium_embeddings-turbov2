#!/usr/bin/env python3
"""
Quick evaluation for Qwen2.5-7B-1024D Model2Vec

Expected performance: 91-95% overall quality
Expected improvement: +7-11% vs Qwen2.5-1.5B baseline (84%)
"""

import numpy as np
from model2vec import StaticModel
from pathlib import Path

print("=" * 80)
print("📊 Qwen2.5-7B-1024D Model2Vec - Quick Evaluation")
print("=" * 80)
print()

# Load model
model_path = "models/qwen25-7b-deposium-1024d"

if not Path(model_path).exists():
    print(f"❌ Model not found: {model_path}")
    print()
    print("Please run distillation first:")
    print("  python3 distill_qwen25_7b.py")
    exit(1)

print(f"📂 Loading model from: {model_path}")
model = StaticModel.from_pretrained(model_path)

test_embedding = model.encode(["test"], show_progress_bar=False)[0]
dimensions = len(test_embedding)

print(f"✅ Model loaded! Dimensions: {dimensions}")
print()


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def evaluate_semantic_similarity():
    """Test semantic similarity understanding"""
    print("1️⃣  Semantic Similarity")
    print("-" * 80)

    test_pairs = [
        ("dog", "puppy", 0.85),
        ("cat", "kitten", 0.85),
        ("car", "automobile", 0.90),
        ("happy", "joyful", 0.80),
        ("big", "large", 0.85),
        ("fast", "quick", 0.85),
        ("computer", "laptop", 0.80),
        ("phone", "smartphone", 0.85),
        # Negative examples
        ("dog", "car", 0.30),
        ("happy", "computer", 0.20),
    ]

    scores = []
    for word1, word2, expected in test_pairs:
        emb1 = model.encode([word1], show_progress_bar=False)[0]
        emb2 = model.encode([word2], show_progress_bar=False)[0]

        similarity = cosine_similarity(emb1, emb2)

        # Score based on how close we are to expected
        error = abs(similarity - expected)
        score = max(0, 1 - error)
        scores.append(score)

        status = "✅" if score > 0.7 else "⚠️"
        print(f"{status} {word1:15s} <-> {word2:15s}: {similarity:.3f} (expected: {expected:.2f}, score: {score:.3f})")

    avg_score = np.mean(scores)
    print(f"\n📊 Semantic Similarity Score: {avg_score:.3f} ({avg_score*100:.1f}%)")
    print()

    return avg_score


def evaluate_instruction_awareness():
    """Test if model understands instructions vs terms"""
    print("2️⃣  Instruction Awareness")
    print("-" * 80)

    instruction_tests = [
        ("Explain machine learning", "machine learning", "query vs topic"),
        ("What is Python?", "Python", "question vs term"),
        ("How to use Docker", "Docker", "instruction vs tool"),
        ("Define recursion", "recursion", "command vs concept"),
        ("Tutorial for Git", "Git", "intent vs subject"),
        ("Compare React vs Vue", "React Vue", "comparison vs terms"),
        ("Debug authentication error", "authentication error", "action vs issue"),
        ("Implement binary search", "binary search", "task vs algorithm"),
    ]

    scores = []
    for instruction, term, description in instruction_tests:
        emb_instruction = model.encode([instruction], show_progress_bar=False)[0]
        emb_term = model.encode([term], show_progress_bar=False)[0]

        similarity = cosine_similarity(emb_instruction, emb_term)

        # For instruction-aware: should be moderate (0.5-0.8), not too high or low
        # Too high (>0.9) = not distinguishing instruction from term
        # Too low (<0.4) = losing semantic connection
        ideal_range = (0.50, 0.85)
        if ideal_range[0] <= similarity <= ideal_range[1]:
            score = 1.0
        elif similarity < ideal_range[0]:
            score = similarity / ideal_range[0]  # penalty for too low
        else:
            score = max(0, 1 - (similarity - ideal_range[1]) * 2)  # penalty for too high

        scores.append(score)

        status = "✅" if score > 0.7 else "⚠️"
        print(f"{status} {description:25s}: sim={similarity:.3f}, score={score:.3f}")

    avg_score = np.mean(scores)
    print(f"\n📊 Instruction Awareness Score: {avg_score:.3f} ({avg_score*100:.1f}%)")
    print()

    return avg_score


def evaluate_code_understanding():
    """Test understanding of code and technical content"""
    print("3️⃣  Code Understanding")
    print("-" * 80)

    code_tests = [
        ("def add(a, b): return a + b", "function to add two numbers", 0.75),
        ("for i in range(10): print(i)", "loop that prints numbers 0 to 9", 0.70),
        ("class User: pass", "user class definition", 0.70),
        ("import numpy as np", "import numpy library", 0.80),
        ("if x > 0: return True", "check if x is positive", 0.75),
        ("try: except Exception: pass", "error handling code", 0.70),
        ("SELECT * FROM users", "get all users from database", 0.75),
        ("docker run -p 8080:80", "run docker container with port mapping", 0.70),
    ]

    scores = []
    for code, description, expected in code_tests:
        emb_code = model.encode([code], show_progress_bar=False)[0]
        emb_desc = model.encode([description], show_progress_bar=False)[0]

        similarity = cosine_similarity(emb_code, emb_desc)

        error = abs(similarity - expected)
        score = max(0, 1 - error)
        scores.append(score)

        status = "✅" if score > 0.6 else "⚠️"
        print(f"{status} sim={similarity:.3f} (exp: {expected:.2f}), score={score:.3f}")
        print(f"    Code: {code[:45]}...")
        print(f"    Desc: {description[:45]}...")

    avg_score = np.mean(scores)
    print(f"\n📊 Code Understanding Score: {avg_score:.3f} ({avg_score*100:.1f}%)")
    print()

    return avg_score


def evaluate_domain_knowledge():
    """Test understanding of domain-specific content"""
    print("4️⃣  Domain Knowledge")
    print("-" * 80)

    domain_tests = [
        ("neural network training", "gradient descent optimization", 0.70),
        ("REST API endpoint", "HTTP GET request", 0.65),
        ("database transaction", "ACID properties", 0.65),
        ("authentication token", "JWT authorization", 0.70),
        ("memory leak", "garbage collection", 0.60),
        ("load balancer", "distribute traffic", 0.65),
        ("CI/CD pipeline", "automated deployment", 0.70),
        ("microservices architecture", "distributed systems", 0.70),
    ]

    scores = []
    for term1, term2, expected in domain_tests:
        emb1 = model.encode([term1], show_progress_bar=False)[0]
        emb2 = model.encode([term2], show_progress_bar=False)[0]

        similarity = cosine_similarity(emb1, emb2)

        error = abs(similarity - expected)
        score = max(0, 1 - error)
        scores.append(score)

        status = "✅" if score > 0.6 else "⚠️"
        print(f"{status} {term1:30s} <-> {term2:30s}")
        print(f"    sim={similarity:.3f} (exp: {expected:.2f}), score={score:.3f}")

    avg_score = np.mean(scores)
    print(f"\n📊 Domain Knowledge Score: {avg_score:.3f} ({avg_score*100:.1f}%)")
    print()

    return avg_score


def evaluate_multilingual():
    """Test multilingual capabilities (Qwen is multilingual)"""
    print("5️⃣  Multilingual Understanding")
    print("-" * 80)

    multilingual_tests = [
        ("Hello", "你好", "English <-> Chinese", 0.65),
        ("machine learning", "機械学習", "EN <-> Japanese", 0.60),
        ("artificial intelligence", "intelligence artificielle", "EN <-> French", 0.70),
        ("computer", "电脑", "EN <-> Chinese", 0.65),
        ("programming", "プログラミング", "EN <-> Japanese", 0.60),
    ]

    scores = []
    for word1, word2, description, expected in multilingual_tests:
        emb1 = model.encode([word1], show_progress_bar=False)[0]
        emb2 = model.encode([word2], show_progress_bar=False)[0]

        similarity = cosine_similarity(emb1, emb2)

        error = abs(similarity - expected)
        score = max(0, 1 - error)
        scores.append(score)

        status = "✅" if score > 0.6 else "⚠️"
        print(f"{status} {description:25s}: sim={similarity:.3f} (exp: {expected:.2f}), score={score:.3f}")

    avg_score = np.mean(scores)
    print(f"\n📊 Multilingual Score: {avg_score:.3f} ({avg_score*100:.1f}%)")
    print()

    return avg_score


def evaluate_context_understanding():
    """Test understanding of context and nuance"""
    print("6️⃣  Context Understanding")
    print("-" * 80)

    context_tests = [
        ("bank account", "financial institution", 0.70),
        ("river bank", "waterside", 0.65),
        ("Python programming", "Python snake", 0.40),  # should distinguish
        ("Apple company", "Apple fruit", 0.40),  # should distinguish
        ("Spring framework", "Spring season", 0.35),  # should distinguish
        ("Java programming", "Java island", 0.40),  # should distinguish
    ]

    scores = []
    for phrase1, phrase2, expected in context_tests:
        emb1 = model.encode([phrase1], show_progress_bar=False)[0]
        emb2 = model.encode([phrase2], show_progress_bar=False)[0]

        similarity = cosine_similarity(emb1, emb2)

        error = abs(similarity - expected)
        score = max(0, 1 - error)
        scores.append(score)

        status = "✅" if score > 0.6 else "⚠️"
        print(f"{status} {phrase1:25s} <-> {phrase2:25s}")
        print(f"    sim={similarity:.3f} (exp: {expected:.2f}), score={score:.3f}")

    avg_score = np.mean(scores)
    print(f"\n📊 Context Understanding Score: {avg_score:.3f} ({avg_score*100:.1f}%)")
    print()

    return avg_score


# Run all evaluations
print("🚀 Running evaluation suite...")
print()

semantic_score = evaluate_semantic_similarity()
instruction_score = evaluate_instruction_awareness()
code_score = evaluate_code_understanding()
domain_score = evaluate_domain_knowledge()
multilingual_score = evaluate_multilingual()
context_score = evaluate_context_understanding()

# Calculate overall score with weights
weights = {
    "semantic": 0.20,
    "instruction": 0.25,  # Most important for our use case
    "code": 0.20,
    "domain": 0.15,
    "multilingual": 0.10,
    "context": 0.10,
}

overall_score = (
    semantic_score * weights["semantic"] +
    instruction_score * weights["instruction"] +
    code_score * weights["code"] +
    domain_score * weights["domain"] +
    multilingual_score * weights["multilingual"] +
    context_score * weights["context"]
)

print("=" * 80)
print("🎯 FINAL RESULTS - Qwen2.5-7B-1024D")
print("=" * 80)
print()

print("Category Scores:")
print(f"  Semantic Similarity:     {semantic_score:.3f} ({semantic_score*100:.1f}%)")
print(f"  Instruction Awareness:   {instruction_score:.3f} ({instruction_score*100:.1f}%) ⭐")
print(f"  Code Understanding:      {code_score:.3f} ({code_score*100:.1f}%)")
print(f"  Domain Knowledge:        {domain_score:.3f} ({domain_score*100:.1f}%)")
print(f"  Multilingual:            {multilingual_score:.3f} ({multilingual_score*100:.1f}%)")
print(f"  Context Understanding:   {context_score:.3f} ({context_score*100:.1f}%)")
print()

print(f"🏆 OVERALL QUALITY: {overall_score:.3f} ({overall_score*100:.1f}%)")
print()

# Comparison with previous model
previous_score = 0.682  # Qwen2.5-1.5B score
improvement = overall_score - previous_score
improvement_pct = (improvement / previous_score) * 100

print("📊 Comparison:")
print(f"  Previous (Qwen2.5-1.5B): {previous_score:.3f} ({previous_score*100:.1f}%)")
print(f"  Current (Qwen2.5-7B):    {overall_score:.3f} ({overall_score*100:.1f}%)")
print(f"  Improvement:             {improvement:+.3f} ({improvement_pct:+.1f}%)")
print()

# Evaluation vs target
target_min = 0.91
target_max = 0.95

if overall_score >= target_min:
    print(f"✅ TARGET ACHIEVED! Score {overall_score*100:.1f}% is within target range (91-95%)")
    print()
    print("🎉 Ready for production deployment!")
    print()
    print("Next steps:")
    print("  1. Update API: Modify api.py to use this model")
    print("  2. Build Docker: docker build -t deposium-embeddings-v11 .")
    print("  3. Deploy and test")
elif overall_score >= target_min - 0.05:
    print(f"⚠️  CLOSE TO TARGET. Score {overall_score*100:.1f}% is slightly below target (91-95%)")
    print()
    print("Consider:")
    print("  - Re-running distillation with larger corpus")
    print("  - Adjusting PCA dimensions")
    print("  - Still acceptable for production")
else:
    print(f"❌ BELOW TARGET. Score {overall_score*100:.1f}% is below target range (91-95%)")
    print()
    print("Troubleshooting:")
    print("  - Check distillation parameters")
    print("  - Verify model loaded correctly")
    print("  - Compare with Qwen2.5-1.5B baseline")

print()
