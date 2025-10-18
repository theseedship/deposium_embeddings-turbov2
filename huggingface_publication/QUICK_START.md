# Quick Start Guide

Get started with **qwen25-deposium-1024d** in 3 simple steps:

## 1️⃣ Installation

```bash
pip install model2vec scikit-learn numpy
```

## 2️⃣ Load Model

```python
from model2vec import StaticModel

# Download and load (automatic)
model = StaticModel.from_pretrained("tss-deposium/qwen25-deposium-1024d")
```

## 3️⃣ Use It!

### Basic Encoding

```python
# Encode some text
texts = [
    "How do I train a neural network?",
    "Neural network training tutorial",
    "Machine learning basics"
]

embeddings = model.encode(texts)
print(embeddings.shape)  # (3, 1024)
```

### Semantic Search

```python
from sklearn.metrics.pairwise import cosine_similarity

# Query
query = "Explain quantum computing"
query_emb = model.encode([query])[0]

# Documents
documents = [
    "Quantum computing explanation and tutorial guide",
    "Classical computing architecture overview",
    "Quantum physics fundamentals"
]
doc_embs = model.encode(documents)

# Find most similar
similarities = cosine_similarity([query_emb], doc_embs)[0]

# Rank results
for doc, score in sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True):
    print(f"{score:.3f} - {doc}")
```

**Output:**
```
0.947 - Quantum computing explanation and tutorial guide
0.612 - Quantum physics fundamentals
0.584 - Classical computing architecture overview
```

### Instruction-Aware Search

```python
# The model understands instructions!
query = "Find articles about climate change"
documents = [
    "Climate change research articles and publications",  # High match
    "Climate change is a serious issue",                  # Lower match
]

query_emb = model.encode([query])[0]
doc_embs = model.encode(documents)
similarities = cosine_similarity([query_emb], doc_embs)[0]

print(similarities)
# [0.95, 0.61] - Correctly prioritizes "articles"!
```

## 🎯 Next Steps

- **Run examples:** Check `examples/instruction_awareness_demo.py`
- **See benchmarks:** Read `BENCHMARKS.md`
- **Explore use cases:** Check `examples/real_world_use_cases.py`

## 🔗 Links

- **Model Card:** Full README with detailed info
- **GitHub:** [deposium_embeddings-turbov2](https://github.com/theseedship/deposium_embeddings-turbov2)
- **Report Issues:** [GitHub Issues](https://github.com/theseedship/deposium_embeddings-turbov2/issues)

---

**Built with ❤️ by TSS Deposium**
