#!/usr/bin/env python3
"""
MTEB Evaluation for Granite 4.0 Micro Model2Vec
Tests compatibility and performance on MTEB quick tasks
"""

import mteb
from model2vec import StaticModel
from pathlib import Path

print("=" * 80)
print("📊 MTEB Evaluation - Granite 4.0 Micro Model2Vec")
print("=" * 80)
print()

# Load model
model_path = Path("granite-4.0-micro-deposium-1024d")

if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    print()
    print("Run distillation first:")
    print("  python3 distill_granite_4_0_micro.py")
    exit(1)

print("📥 Loading Granite 4.0 Micro Model2Vec...")
try:
    model = StaticModel.from_pretrained(str(model_path))
    print("✅ Model loaded successfully")
    print()
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Get model dimensions
print("🔍 Detecting model dimensions...")
test_embedding = model.encode(["test"])[0]
dimensions = len(test_embedding)
print(f"✅ Dimensions: {dimensions}")
print()

# Create wrapper for MTEB
class Model2VecWrapper:
    def __init__(self, model):
        self.model = model

    def encode(self, sentences, **kwargs):
        # MTEB expects encode() to return numpy array
        return self.model.encode(sentences)

    @property
    def embedding_size(self):
        # MTEB requires embedding_size property
        test_emb = self.model.encode(["test"])[0]
        return len(test_emb)

print("🔧 Creating MTEB wrapper...")
wrapped_model = Model2VecWrapper(model)
print("✅ Wrapper ready")
print()

# MTEB quick tasks (fast evaluation)
print("=" * 80)
print("📋 MTEB Quick Tasks (7 tasks)")
print("=" * 80)
print()

quick_tasks = [
    "Banking77Classification",
    "EmotionClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "STS12",
]

print("⚠️  Note: Model2Vec typically completes 3-4/7 MTEB tasks")
print("   Some tasks require features not implemented by Model2Vec")
print()

# Run evaluation
results_dir = Path("mteb_results_granite")
results_dir.mkdir(parents=True, exist_ok=True)

print(f"📂 Results will be saved to: {results_dir}/")
print()

completed_tasks = 0
failed_tasks = 0

for task_name in quick_tasks:
    print(f"🔄 Running: {task_name}")

    try:
        # Load task
        task = mteb.get_task(task_name)

        # Run evaluation
        evaluation = mteb.MTEB(tasks=[task])
        results = evaluation.run(
            wrapped_model,
            output_folder=str(results_dir),
            verbosity=0,
        )

        print(f"   ✅ Completed")
        completed_tasks += 1

    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:100]}")
        failed_tasks += 1

    print()

print("=" * 80)
print("📊 MTEB EVALUATION SUMMARY")
print("=" * 80)
print()

print(f"✅ Completed: {completed_tasks}/{len(quick_tasks)} tasks")
print(f"❌ Failed: {failed_tasks}/{len(quick_tasks)} tasks")
print()

if completed_tasks > 0:
    print(f"📂 Results saved in: {results_dir}/")
    print()
    print("View results with:")
    print(f"  python3 show_mteb_results.py {results_dir}")
else:
    print("⚠️  No tasks completed successfully")
    print()
    print("This is expected for Model2Vec models.")
    print("Use custom evaluation instead:")
    print("  python3 compare_all_models_v2.py")
    print("  python3 test_multilingual_granite.py")

print()
print("=" * 80)
print("💡 Recommendation")
print("=" * 80)
print()

print("MTEB is designed for full transformer models, not static embeddings.")
print()
print("For Model2Vec evaluation, use:")
print("  1. compare_all_models_v2.py - Quality, speed, instruction-awareness")
print("  2. test_multilingual_granite.py - Multilingual capabilities")
print("  3. Custom benchmarks matching your specific use case")
print()

# Compare with baseline if available
baseline_dir = Path("mteb_results_baseline/sentence-transformers__all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf")

if baseline_dir.exists():
    import json

    print("=" * 80)
    print("📊 Comparison with Baseline (if available)")
    print("=" * 80)
    print()

    # Try to load and compare results
    granite_results = {}
    baseline_results = {}

    # Load Granite results
    granite_model_dir = list(results_dir.glob("*"))
    if granite_model_dir:
        for json_file in granite_model_dir[0].glob("*.json"):
            if json_file.name != "model_meta.json":
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        if "scores" in data and "test" in data["scores"]:
                            task_name = json_file.stem
                            score = data["scores"]["test"][0].get("main_score")
                            if score is not None:
                                granite_results[task_name] = score
                except:
                    pass

    # Load baseline results
    for json_file in baseline_dir.glob("*.json"):
        if json_file.name != "model_meta.json":
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    if "scores" in data and "test" in data["scores"]:
                        task_name = json_file.stem
                        score = data["scores"]["test"][0].get("main_score")
                        if score is not None:
                            baseline_results[task_name] = score
            except:
                pass

    if granite_results and baseline_results:
        print(f"{'Task':<40} {'Baseline':<15} {'Granite':<15} {'Diff':<10}")
        print("-" * 80)

        for task in sorted(set(granite_results.keys()) & set(baseline_results.keys())):
            baseline_score = baseline_results[task] * 100
            granite_score = granite_results[task] * 100
            diff = granite_score - baseline_score

            print(f"{task:<40} {baseline_score:>6.2f}%        {granite_score:>6.2f}%        {diff:>+6.2f}%")

        print()

        # Average
        if granite_results:
            granite_avg = sum(granite_results.values()) / len(granite_results) * 100
            baseline_avg = sum(baseline_results.values()) / len(baseline_results) * 100

            print(f"Average:")
            print(f"  Baseline: {baseline_avg:.2f}%")
            print(f"  Granite:  {granite_avg:.2f}%")
            print(f"  Diff:     {granite_avg - baseline_avg:+.2f}%")

print()
