#!/usr/bin/env python3
"""
Test Simple: mxbai-edge-colbert-v0-32m
Architecture: ColBERT (multi-vector, late interaction)

Ce script teste le modèle ColBERT pour évaluer:
1. Qualité des embeddings (similarité sémantique)
2. Instruction-awareness (vs qwen25: 94.9%)
3. Code understanding (vs qwen25: 84.5%)
4. Performance (RAM, vitesse)
"""

import time
import psutil
import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_memory_usage():
    """Get current process memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def maxsim_score(query_emb, doc_emb):
    """
    Calculate MaxSim score between query and document embeddings

    MaxSim operation:
    - For each query token embedding, find max similarity with all doc token embeddings
    - Average these max similarities

    Args:
        query_emb: numpy array or tensor of shape (num_query_tokens, embedding_dim)
        doc_emb: numpy array or tensor of shape (num_doc_tokens, embedding_dim)

    Returns:
        float: MaxSim score
    """
    import torch

    # Convert numpy arrays to torch tensors if needed
    if not isinstance(query_emb, torch.Tensor):
        query_emb = torch.from_numpy(query_emb)
    if not isinstance(doc_emb, torch.Tensor):
        doc_emb = torch.from_numpy(doc_emb)

    # Normalize embeddings
    query_emb = query_emb / query_emb.norm(dim=1, keepdim=True)
    doc_emb = doc_emb / doc_emb.norm(dim=1, keepdim=True)

    # Compute similarity matrix: (num_query_tokens, num_doc_tokens)
    sim_matrix = torch.matmul(query_emb, doc_emb.T)

    # For each query token, take max similarity across all doc tokens
    max_sims, _ = torch.max(sim_matrix, dim=1)

    # Average across query tokens
    score = torch.mean(max_sims).item()

    return score


def main():
    logger.info("=" * 80)
    logger.info("🧪 Testing mxbai-edge-colbert-v0-32m (ColBERT Architecture)")
    logger.info("=" * 80)

    # Measure initial memory
    mem_before = get_memory_usage()
    logger.info(f"\n📊 Initial memory: {mem_before:.1f} MB")

    # Import and load model
    logger.info("\n📥 Loading ColBERT model from HuggingFace...")
    try:
        from pylate import models
        import torch
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("   Install with: pip install pylate torch")
        return False

    start_time = time.time()

    # Load model
    model = models.ColBERT(
        model_name_or_path="mixedbread-ai/mxbai-edge-colbert-v0-32m",
    )

    load_time = time.time() - start_time
    mem_after_load = get_memory_usage()
    model_size = mem_after_load - mem_before

    logger.info(f"✅ Model loaded in {load_time:.2f}s")
    logger.info(f"   Model size in RAM: {model_size:.1f} MB")
    logger.info(f"   Total memory: {mem_after_load:.1f} MB")

    # ============================================================================
    # Test 1: Semantic Similarity (Baseline)
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info("🔬 Test 1: Semantic Similarity (Baseline)")
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
    logger.info("\n📊 Similar pairs:")
    for s1, s2 in similar_pairs:
        queries_embeddings = model.encode([s1], is_query=True)
        documents_embeddings = model.encode([s2], is_query=False)

        # ColBERT uses MaxSim operation
        score = maxsim_score(
            queries_embeddings[0],
            documents_embeddings[0]
        )
        similar_scores.append(score)
        logger.info(f"   {score:.4f} - '{s1[:40]}...' ↔ '{s2[:40]}...'")

    dissimilar_scores = []
    logger.info("\n📊 Dissimilar pairs:")
    for s1, s2 in dissimilar_pairs:
        queries_embeddings = model.encode([s1], is_query=True)
        documents_embeddings = model.encode([s2], is_query=False)

        score = maxsim_score(
            queries_embeddings[0],
            documents_embeddings[0]
        )

        dissimilar_scores.append(score)
        logger.info(f"   {score:.4f} - '{s1[:40]}...' ↔ '{s2[:40]}...'")

    avg_similar = sum(similar_scores) / len(similar_scores)
    avg_dissimilar = sum(dissimilar_scores) / len(dissimilar_scores)
    separation = avg_similar - avg_dissimilar

    logger.info(f"\n✅ Semantic Similarity Results:")
    logger.info(f"   Similar pairs avg: {avg_similar:.4f}")
    logger.info(f"   Dissimilar pairs avg: {avg_dissimilar:.4f}")
    logger.info(f"   Separation: {separation:.4f} (higher = better)")

    semantic_score = avg_similar

    # ============================================================================
    # Test 2: Instruction Awareness (vs qwen25: 94.9%)
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info("🎯 Test 2: Instruction Awareness (UNIQUE to qwen25)")
    logger.info(f"{'='*80}")

    instruction_pairs = [
        ("How do I install Python?", "What are the steps to set up Python on my computer?"),
        ("Can you explain recursion?", "Please describe how recursive functions work"),
        ("Show me how to sort a list", "Demonstrate list sorting in Python"),
        ("What is the capital of France?", "Tell me which city is the capital of France"),
    ]

    instruction_scores = []
    logger.info("\n📊 Instruction pairs:")
    for s1, s2 in instruction_pairs:
        queries_embeddings = model.encode([s1], is_query=True)
        documents_embeddings = model.encode([s2], is_query=False)

        score = maxsim_score(
            queries_embeddings[0],
            documents_embeddings[0]
        )

        instruction_scores.append(score)
        logger.info(f"   {score:.4f} - '{s1}' ↔ '{s2}'")

    avg_instruction = sum(instruction_scores) / len(instruction_scores)
    logger.info(f"\n✅ Instruction Awareness: {avg_instruction:.4f}")
    logger.info(f"   vs qwen25 (0.949): {'+' if avg_instruction > 0.949 else ''}{((avg_instruction - 0.949) * 100):.1f}%")

    # ============================================================================
    # Test 3: Code Understanding (vs qwen25: 84.5%)
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info("💻 Test 3: Code Understanding")
    logger.info(f"{'='*80}")

    code_pairs = [
        ("def hello(): print('hi')", "A function that prints hello"),
        ("for i in range(10): print(i)", "Loop that prints numbers 0 to 9"),
        ("import numpy as np", "Import the NumPy library for numerical computing"),
        ("class Dog: pass", "Define an empty Dog class"),
    ]

    code_scores = []
    logger.info("\n📊 Code pairs:")
    for code, desc in code_pairs:
        queries_embeddings = model.encode([code], is_query=True)
        documents_embeddings = model.encode([desc], is_query=False)

        score = maxsim_score(
            queries_embeddings[0],
            documents_embeddings[0]
        )

        code_scores.append(score)
        logger.info(f"   {score:.4f} - '{code}' ↔ '{desc}'")

    avg_code = sum(code_scores) / len(code_scores)
    logger.info(f"\n✅ Code Understanding: {avg_code:.4f}")
    logger.info(f"   vs qwen25 (0.845): {'+' if avg_code > 0.845 else ''}{((avg_code - 0.845) * 100):.1f}%")

    # ============================================================================
    # Test 4: Performance Metrics
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info("⚡ Test 4: Performance Metrics")
    logger.info(f"{'='*80}")

    # Measure encoding speed
    test_texts = [
        "This is a test sentence for measuring encoding speed.",
        "Machine learning models process text efficiently.",
        "Embeddings capture semantic meaning of text."
    ]

    # Warmup
    _ = model.encode(test_texts, is_query=True)

    # Measure
    start = time.time()
    for _ in range(10):
        _ = model.encode(test_texts, is_query=True)
    avg_time = (time.time() - start) / 10 * 1000  # ms

    logger.info(f"\n📊 Performance:")
    logger.info(f"   Encoding speed: {avg_time:.2f} ms for {len(test_texts)} texts")
    logger.info(f"   Per text: {avg_time / len(test_texts):.2f} ms")
    logger.info(f"   Model RAM: {model_size:.1f} MB")
    logger.info(f"   Total RAM: {get_memory_usage():.1f} MB")

    # ============================================================================
    # Final Results
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info("📊 FINAL RESULTS SUMMARY")
    logger.info(f"{'='*80}")

    overall_quality = (semantic_score + avg_instruction + avg_code) / 3

    logger.info(f"\n🎯 Quality Metrics:")
    logger.info(f"   Overall Quality: {overall_quality:.4f} ({overall_quality * 100:.1f}%)")
    logger.info(f"   Semantic Similarity: {semantic_score:.4f}")
    logger.info(f"   Instruction Awareness: {avg_instruction:.4f} (qwen25: 0.949)")
    logger.info(f"   Code Understanding: {avg_code:.4f} (qwen25: 0.845)")

    logger.info(f"\n⚡ Performance:")
    logger.info(f"   Model Size: {model_size:.1f} MB")
    logger.info(f"   Encoding Speed: {avg_time / len(test_texts):.2f} ms/text")
    logger.info(f"   RAM Overhead: +{model_size:.0f} MB vs current")

    logger.info(f"\n🤔 Recommendation:")

    # Decision criteria
    better_than_qwen25 = overall_quality > 0.682  # qwen25 overall quality
    acceptable_ram = model_size < 1000  # < 1GB
    acceptable_speed = (avg_time / len(test_texts)) < 100  # < 100ms

    if better_than_qwen25 and acceptable_ram and acceptable_speed:
        recommendation = "✅ WORTH INTEGRATION - Superior quality with acceptable performance"
    elif better_than_qwen25:
        recommendation = "⚠️ CONSIDER - Better quality but performance concerns"
    else:
        recommendation = "❌ TEST ONLY - Not better than current qwen25-1024d"

    logger.info(f"   {recommendation}")

    # Save results to file
    results_file = Path("results.txt")
    with open(results_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("mxbai-edge-colbert-v0-32m Test Results\n")
        f.write("=" * 80 + "\n\n")
        f.write("Architecture: ColBERT (multi-vector, late interaction)\n")
        f.write("Parameters: 32M\n")
        f.write(f"Model Size: {model_size:.1f} MB\n")
        f.write(f"RAM Usage: {get_memory_usage():.1f} MB total\n\n")

        f.write("=" * 80 + "\n")
        f.write("Similarity Tests (MaxSim scores)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Semantic Similarity: {semantic_score:.4f}\n")
        f.write(f"- Similar pairs avg: {avg_similar:.4f}\n")
        f.write(f"- Dissimilar pairs avg: {avg_dissimilar:.4f}\n")
        f.write(f"- Separation: {separation:.4f}\n\n")

        f.write(f"Instruction Awareness: {avg_instruction:.4f}\n")
        f.write(f"- vs qwen25 (0.949): {'+' if avg_instruction > 0.949 else ''}{((avg_instruction - 0.949) * 100):.1f}%\n\n")

        f.write(f"Code Understanding: {avg_code:.4f}\n")
        f.write(f"- vs qwen25 (0.845): {'+' if avg_code > 0.845 else ''}{((avg_code - 0.845) * 100):.1f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write("Performance\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Encoding speed: {avg_time / len(test_texts):.2f} ms/text\n")
        f.write(f"Model RAM: {model_size:.1f} MB\n")
        f.write(f"RAM overhead vs current: +{model_size:.0f} MB\n\n")

        f.write("=" * 80 + "\n")
        f.write("Recommendation\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{recommendation}\n\n")

        f.write(f"Overall Quality: {overall_quality:.4f} ({overall_quality * 100:.1f}%)\n")
        f.write(f"vs qwen25-1024d (68.2%): {'+' if overall_quality > 0.682 else ''}{((overall_quality - 0.682) * 100):.1f}%\n")

    logger.info(f"\n💾 Results saved to: {results_file}")
    logger.info("\n✅ Test complete!")

    return True


if __name__ == "__main__":
    main()
