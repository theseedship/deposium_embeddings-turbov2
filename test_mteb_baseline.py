#!/usr/bin/env python3
"""
Test MTEB with a baseline sentence-transformers model for comparison
"""

import mteb
from sentence_transformers import SentenceTransformer
from datetime import datetime

print("=" * 80)
print("MTEB Baseline Test - sentence-transformers/all-MiniLM-L6-v2")
print("=" * 80)
print()

# Load model
print("Loading baseline model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Model loaded!")
print()

# Define same quick tasks
quick_tasks = [
    "Banking77Classification",
    "TwentyNewsgroupsClustering",
    "SprintDuplicateQuestions",
    "NFCorpus",
    "SciFact",
    "STSBenchmark",
    "SICK-R",
]

print(f"Running {len(quick_tasks)} tasks...")
print()

# Create MTEB evaluation
evaluation = mteb.MTEB(tasks=mteb.get_tasks(tasks=quick_tasks, languages=["en"]))

# Run evaluation
start_time = datetime.now()

results = evaluation.run(
    model,
    output_folder="mteb_results_baseline",
    eval_splits=["test"],
    overwrite_results=False,
)

end_time = datetime.now()
duration = end_time - start_time

print()
print("=" * 80)
print("✅ Baseline Evaluation Complete!")
print("=" * 80)
print(f"Duration: {duration}")
print()

# Calculate average (if results is a list)
if isinstance(results, list):
    print("✅ All tasks completed successfully!")
    print(f"Number of results: {len(results)}")
else:
    print("Results type:", type(results))
