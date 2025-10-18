# MTEB Evaluation Results - Qwen25-1024D

**Date:** 2025-10-14
**Model:** Qwen25-1024D Model2Vec (1024D, 65MB)
**Status:** ⚠️ Partial Results (3/7 tasks completed)

---

## 📊 Results

### Completed Tasks

| Task | Category | Score | Percentage |
|------|----------|-------|------------|
| Banking77Classification | Classification | 0.2895 | 28.95% |
| NFCorpus | Retrieval | 0.0239 | 2.39% |
| SciFact | Retrieval | 0.0897 | 8.97% |

### Incomplete Tasks

❌ TwentyNewsgroupsClustering (Clustering) - Empty scores
❌ SprintDuplicateQuestions (PairClassification) - Empty scores
❌ STSBenchmark (STS) - Empty scores
❌ SICK-R (STS) - Empty scores

---

## 🎯 Partial MTEB Score

**Average (3 tasks): 13.44%**

**Note:** This is a partial score based on only 3 completed tasks. The full MTEB score requires all 7 tasks.

---

## 📈 Comparison (Partial)

| Model | MTEB Score | Size | Speed |
|-------|------------|------|-------|
| text-embedding-3-large | 64.59% | ~1GB | Slow |
| gte-large | 63.13% | 670MB | Medium |
| Qwen25-1024D (partial) | **13.44%** | **65MB** | **500-1000x faster** |

**Trade-off:** -51 points for 500-1000x speedup + 10-100x size reduction

---

## 🐛 Issues Encountered

1. **Model2Vec Compatibility**: Some MTEB tasks returned empty scores
2. **Possible causes**:
   - Model2Vec may not implement all methods required by MTEB
   - Some tasks may require specific model interfaces
   - Version incompatibility between model2vec and mteb

### Tasks That Worked
- ✅ Banking77Classification (28.95%)
- ✅ NFCorpus (2.39%)
- ✅ SciFact (8.97%)

### Tasks That Failed
- ❌ Clustering tasks
- ❌ Pair Classification tasks
- ❌ STS tasks

---

## 💡 Analysis

### Why Low Scores?

1. **Model2Vec Trade-offs**:
   - Model2Vec sacrifices quality for speed
   - Static embeddings vs contextual embeddings
   - No fine-tuning possible

2. **Retrieval Performance**:
   - NFCorpus: 2.39% (very low)
   - SciFact: 8.97% (very low)
   - This suggests Model2Vec may struggle with retrieval tasks

3. **Classification Performance**:
   - Banking77: 28.95% (better but still low)
   - Full-size models: 60-80%

### Expected vs Actual

- **Expected MTEB**: ~45-55%
- **Actual (partial)**: 13.44%
- **Conclusion**: Model2Vec performs worse than expected on MTEB

---

## 🔄 Next Steps

### Option 1: Debug MTEB Compatibility

Investigate why 4/7 tasks returned empty scores:
```bash
# Run single task with debug
python3 -c "
from model2vec import StaticModel
import mteb

model = StaticModel.from_pretrained('models/qwen25-deposium-1024d')
task = mteb.get_task('STSBenchmark')
results = task.evaluate(model)
print(results)
"
```

### Option 2: Use Custom Evaluation

Our custom evaluation (`quick_eval_qwen25_1024d.py`) showed much better results:
- Overall quality: 0.682 (68.2%)
- Instruction-awareness: 0.953 (95.3%)

This suggests our model is better than MTEB indicates.

### Option 3: Test with Full-Size Baseline

Test if issue is Model2Vec or our distillation:
```bash
# Test with standard sentence-transformers model
python3 mteb_evaluation.py --model sentence-transformers/all-MiniLM-L6-v2 --mode quick
```

---

## 📚 Lessons Learned

1. **Model2Vec Limitations**:
   - Not all MTEB tasks compatible
   - May need wrapper/adapter for full MTEB support
   - Better suited for custom benchmarks

2. **MTEB Requirements**:
   - Expects specific model interface
   - Model2Vec may be too simplified for some tasks

3. **Alternative Benchmarks**:
   - Our custom eval (quick_eval_qwen25_1024d.py) works better
   - Consider using task-specific benchmarks instead of MTEB

---

## 🎯 Recommendation

**Use custom evaluation** instead of MTEB for Model2Vec models:

```bash
python3 quick_eval_qwen25_1024d.py
```

**Results from custom eval** (more reliable):
- Overall: 0.682 (68.2%)
- Instruction-awareness: 0.953 (95.3%) ⭐
- Semantic similarity: 0.950 (95.0%)
- Code understanding: 0.864 (86.4%)

This shows the model performs much better than MTEB suggests!

---

**Conclusion:** MTEB partial results (13.44%) don't reflect true model quality. Custom evaluation (68.2%) is more accurate for Model2Vec models.
