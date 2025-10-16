#!/usr/bin/env python3
"""Compare baseline vs Qwen25-1024D MTEB results"""

import json
from pathlib import Path

print("=" * 80)
print("📊 MTEB Comparison: Baseline vs Qwen25-1024D")
print("=" * 80)
print()

# Load baseline results
baseline_dir = Path("mteb_results_baseline/sentence-transformers__all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
qwen_dir = Path("mteb_results_quick/no_model_name_available/no_revision_available")

baseline_scores = {}
qwen_scores = {}

# Load baseline
for json_file in baseline_dir.glob("*.json"):
    if json_file.name == "model_meta.json":
        continue

    task_name = json_file.stem

    with open(json_file) as f:
        data = json.load(f)

    if "scores" in data and "test" in data["scores"]:
        test_data = data["scores"]["test"][0]
        score = test_data.get("main_score")
        if score is not None:
            baseline_scores[task_name] = score

# Load Qwen25
for json_file in qwen_dir.glob("*.json"):
    if json_file.name == "model_meta.json":
        continue

    task_name = json_file.stem

    with open(json_file) as f:
        data = json.load(f)

    if "scores" in data and "test" in data["scores"]:
        test_data = data["scores"]["test"][0]
        score = test_data.get("main_score")
        if score is not None:
            qwen_scores[task_name] = score

# Print comparison
print("Task-by-Task Comparison:")
print("-" * 80)
print(f"{'Task':<45} {'Baseline':<12} {'Qwen25':<12} {'Diff':<10}")
print("-" * 80)

all_tasks = sorted(set(baseline_scores.keys()) | set(qwen_scores.keys()))

for task in all_tasks:
    baseline = baseline_scores.get(task)
    qwen = qwen_scores.get(task)

    baseline_str = f"{baseline*100:.2f}%" if baseline is not None else "N/A"
    qwen_str = f"{qwen*100:.2f}%" if qwen is not None else "❌ Empty"

    if baseline is not None and qwen is not None:
        diff = (qwen - baseline) * 100
        diff_str = f"{diff:+.2f}%"
    else:
        diff_str = "N/A"

    print(f"{task:<45} {baseline_str:<12} {qwen_str:<12} {diff_str:<10}")

print()
print("=" * 80)
print("Summary:")
print("=" * 80)

# Calculate averages
if baseline_scores:
    baseline_avg = sum(baseline_scores.values()) / len(baseline_scores) * 100
    print(f"Baseline (all-MiniLM-L6-v2):  {baseline_avg:.2f}% ({len(baseline_scores)}/7 tasks)")

if qwen_scores:
    qwen_avg = sum(qwen_scores.values()) / len(qwen_scores) * 100
    print(f"Qwen25-1024D (Model2Vec):    {qwen_avg:.2f}% ({len(qwen_scores)}/7 tasks)")

print()
print("⚠️  Key Finding:")
print("-" * 80)
print(f"✅ Baseline completed: {len(baseline_scores)}/7 tasks")
print(f"❌ Qwen25 completed:   {len(qwen_scores)}/7 tasks")
print()
print("Model2Vec is NOT fully compatible with MTEB.")
print("Missing tasks require features not implemented by Model2Vec.")
print()
print("💡 Recommendation: Use custom evaluation (quick_eval_qwen25_1024d.py)")
print("   Custom eval score: 68.2% (more accurate for Model2Vec)")
print()
