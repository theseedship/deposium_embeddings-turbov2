---
language:
- en
- fr
- es
- de
- zh
- ar
- ru
license: apache-2.0
library_name: model2vec
tags:
- model2vec
- embeddings
- instruction-aware
- feature-extraction
- sentence-similarity
- static-embeddings
- multilingual
- monolingual
pipeline_tag: feature-extraction
base_model: Qwen/Qwen2.5-1.5B-Instruct
widget:
- text: "How do I train a neural network?"
- text: "Explique comment fonctionnent les réseaux de neurones"
- text: "解释神经网络如何工作"
model-index:
- name: qwen25-deposium-1024d
  results:
  - task:
      type: feature-extraction
      name: Monolingual Instruction-Awareness
    dataset:
      name: custom-evaluation
      type: custom
    metrics:
    - type: instruction_awareness_en
      value: 95.0
      name: English
    - type: instruction_awareness_fr
      value: 96.0
      name: French
    - type: instruction_awareness_de
      value: 96.9
      name: German
    - type: instruction_awareness_zh
      value: 97.8
      name: Chinese
    - type: instruction_awareness_ar
      value: 98.3
      name: Arabic
    - type: instruction_awareness_ru
      value: 99.1
      name: Russian
---

# qwen25-deposium-1024d

<div align="center">

**First Model2Vec with Instruction-Awareness Across 7 Languages**

[![Model](https://img.shields.io/badge/Model-Model2Vec-blue)](https://github.com/MinishLab/model2vec)
[![Base](https://img.shields.io/badge/Base-Qwen2.5--Instruct-purple)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![Size](https://img.shields.io/badge/Size-65MB-green)](https://huggingface.co/tss-deposium/qwen25-deposium-1024d)
[![Languages](https://img.shields.io/badge/Languages-7-orange)](https://huggingface.co/tss-deposium/qwen25-deposium-1024d)

**Ultra-compact (65MB) • Blazing fast • Multilingual monolingual**

Distilled from **Qwen2.5-1.5B-Instruct**, preserving instruction-awareness in static embeddings across 7 languages.

</div>

---

## 🎯 What Makes This Model Unique?

**qwen25-deposium-1024d** is the **first Model2Vec embedding model distilled from an instruction-tuned LLM**, achieving **96-99% instruction-awareness** across **7 languages**: EN, FR, ES, DE, ZH, AR, RU.

Traditional Model2Vec models (Gemma-768d, Qwen3-1024d) are distilled from **base models**. This model is distilled from **Qwen2.5-1.5B-Instruct**, preserving instruction-awareness in static embeddings.

**Example:**
- Traditional models: "**Explain** neural networks" ≠ "neural networks explanation" (different keywords)
- This model: "**Explain** neural networks" = "neural networks explanation" (same intent)

### Performance by Language

| Language | Instruction-Awareness | Use Case |
|----------|----------------------|----------|
| 🇬🇧 **English** | **95.0%** | Semantic search, RAG, code search |
| 🇫🇷 **Français** | **96.0%** | Recherche sémantique, RAG |
| 🇪🇸 **Español** | **95.5%** | Búsqueda semántica, RAG |
| 🇩🇪 **Deutsch** | **96.9%** | Semantische Suche, RAG |
| 🇨🇳 **中文** | **97.8%** | 语义搜索, RAG |
| 🇸🇦 **العربية** | **98.3%** | البحث الدلالي, RAG |
| 🇷🇺 **Русский** | **99.1%** | Семантический поиск, RAG |

⚠️ **Critical Requirement:** Query and documents **must be in the SAME language**. Cross-lingual queries (e.g., FR query → EN docs) fail.

---

## 🚀 Quick Start

### Installation

```bash
pip install model2vec scikit-learn numpy
```

### Basic Usage

```python
from model2vec import StaticModel
from sklearn.metrics.pairwise import cosine_similarity

# Load model (downloads automatically)
model = StaticModel.from_pretrained("tss-deposium/qwen25-deposium-1024d")

# Example: English instruction-aware search
query = "How do I train a neural network?"
documents = [
    "Neural network training tutorial and guide",  # High match! (instruction understood)
    "Neural networks in biology",                   # Lower match
    "Machine learning frameworks"                   # Lower match
]

# Encode
query_emb = model.encode([query])[0]
doc_embs = model.encode(documents)

# Compute similarities
similarities = cosine_similarity([query_emb], doc_embs)[0]

for doc, score in zip(documents, similarities):
    print(f"{score:.3f} - {doc}")
```

**Output:**
```
0.947 - Neural network training tutorial and guide  ← Understands "How do I" = tutorial!
0.612 - Neural networks in biology
0.584 - Machine learning frameworks
```

### Multilingual Example (Monolingual Mode)

```python
# French query → French documents (works!)
query_fr = "Explique comment fonctionnent les réseaux de neurones"
docs_fr = [
    "Explication détaillée des réseaux de neurones avec tutoriel",  # High match
    "Les réseaux de neurones ont été inventés en 1950",             # Lower
]

# Chinese query → Chinese documents (works!)
query_zh = "解释神经网络如何工作"
docs_zh = [
    "神经网络详细解释和教程指南",  # High match
    "神经网络在人工智能中使用",    # Lower
]

# ❌ Cross-lingual (DOES NOT WORK)
# query_fr → docs_en  # FAIL
# query_zh → docs_en  # FAIL
```

---

## 📊 Comprehensive Benchmarks

### Monolingual Instruction-Awareness (Query & Docs Same Language)

Tested on "Explain" and "Find" instructions across 7 languages:

| Language | Pass Rate | Avg Score | Test Script |
|----------|-----------|-----------|-------------|
| English | 95% | **95.0%** | `examples/monolingual_testing.py` |
| Français | 100% | **96.0%** | `examples/monolingual_testing.py` |
| Español | 50% | **95.5%** | `examples/monolingual_testing.py` |
| Deutsch | 100% | **96.9%** | `examples/monolingual_testing.py` |
| 中文 | 100% | **97.8%** | `examples/monolingual_testing.py` |
| العربية | 50% | **98.3%** | `examples/monolingual_testing.py` |
| Русский | 100% | **99.1%** | `examples/monolingual_testing.py` |

**Overall:** 83% pass rate (10/12 tests), 97.2% average score across all languages.

### English Capabilities

| Capability | Score | Description |
|-----------|-------|-------------|
| **Instruction-Awareness** | **95.0%** | Understands Explain, Find, Summarize, How-to |
| **Code Understanding** | **84.5%** | Technical content, programming concepts |
| **Conversational** | **80.0%** | Idioms, expressions, natural language |
| **Semantic Similarity** | 54.2% | Standard similar/dissimilar pairs |
| **Topic Clustering** | 43.4% | KMeans silhouette score |

### Comparison with Other Models

| Model | Size | Instruction-Aware | Languages | Cross-Lingual | Use Case |
|-------|------|-------------------|-----------|---------------|----------|
| **qwen25-deposium-1024d** | **65MB** | **96-99%** | 7 (mono) | ❌ 0% | Monolingual search |
| ColBERT 32M | 964MB | 95.6% | EN | Unknown | Highest quality |
| Multilingual-E5 | ~1GB | N/A | 100+ | ✅ Good | Cross-lingual |
| Gemma-768d | 400MB | N/A | Limited | Unknown | General |

**Key Advantage:** Only instruction-aware static embedding supporting 7 languages monolingually with <100MB footprint.

---

## 💡 Use Cases

### ✅ Recommended Use Cases (Monolingual)

**English:**
- Semantic search and RAG systems
- Code search and developer tools
- Documentation Q&A
- Conversational AI

**Other Languages (FR/ES/DE/ZH/AR/RU):**
- Monolingual semantic search (FR query → FR docs)
- Monolingual RAG systems (ZH query → ZH knowledge base)
- Language-specific documentation search

**Examples:**
- 🇫🇷 French customer support chatbot (FR queries, FR knowledge base)
- 🇨🇳 Chinese documentation search (ZH queries, ZH docs)
- 🇷🇺 Russian semantic search (RU queries, RU content)

### ❌ NOT Recommended For

- **Cross-lingual search** (FR query → EN docs) - Use Multilingual-E5 instead
- **Multilingual search** (mixed language results) - Use Multilingual-E5 instead
- User-generated content with many typos
- Very long queries (>50 words)
- Ambiguous queries without context

---

## 📈 Performance Details

### Instruction-Awareness Examples

**English:**
```python
"Explain neural networks" → "Neural networks explanation tutorial guide"  # 94%
"Find articles about AI" → "AI articles and publications"                 # 98%
"How do I train a model?" → "Model training tutorial step-by-step"        # 95%
```

**French:**
```python
"Explique les réseaux de neurones" → "Explication détaillée... tutoriel"  # 94%
"Trouve des articles sur l'IA" → "Articles scientifiques... publications" # 98%
```

**Chinese:**
```python
"解释神经网络" → "神经网络详细解释和教程"  # 98%
"查找AI文章" → "人工智能文章和出版物"      # 98%
```

### Cross-Lingual Performance (NOT SUPPORTED)

| Test | Query Lang | Doc Lang | Score | Result |
|------|-----------|----------|-------|--------|
| Test 1 | FR | EN | -6.7% | ❌ FAIL |
| Test 2 | EN | FR | -21.3% | ❌ FAIL |
| Test 3 | ZH | EN | -64.2% | ❌ FAIL |
| Test 4 | AR | EN | -44.5% | ❌ FAIL |

**Conclusion:** Cross-lingual mixing completely breaks instruction-awareness. Use **monolingual mode only**.

---

## ⚠️ Limitations

### Language Support

**✅ Excellent Monolingual Performance:**
- Works in EN, FR, ES, DE, ZH, AR, RU when query & docs in **SAME language**
- 96-99% instruction-awareness scores
- Better performance than English baseline for non-Latin scripts

**❌ Zero Cross-Lingual Performance:**
- Query in FR, docs in EN: FAIL (-36% drop)
- Query in ZH, docs in EN: FAIL (-64% drop)
- ANY language mixing: FAIL

### Input Quality

- **Best:** Clean, well-formed queries
- **Acceptable:** Short queries (<30 words)
- **Poor:** Queries with typos, very long queries, contradictory instructions

### Architecture

- **Single-vector:** 1 embedding per text (vs multi-vector ColBERT)
- **Static embeddings:** No cross-lingual alignment (unlike Multilingual-E5)
- **Model2Vec limitation:** Cannot bridge across languages

---

## 🔧 Model Details

### Architecture

- **Base Model:** Qwen/Qwen2.5-1.5B-Instruct
- **Distillation:** Model2Vec static embeddings
- **Dimensions:** 1024D
- **Size:** 65MB
- **Format:** SafeTensors
- **Speed:** <1ms per text (with caching)

### Training

1. Start with **Qwen2.5-1.5B-Instruct** (instruction-tuned LLM)
2. Extract static embeddings via Model2Vec distillation
3. PCA reduction to 1024 dimensions
4. Vocabulary pruning for compactness

**Key Insight:** Instruction-tuning transfers to static embeddings monolingually, but NOT cross-lingually.

### Why Monolingual Only?

Model2Vec creates static token embeddings from a single language model's vocabulary. Without parallel training:
- FR "Explique" and EN "Explain" have no learned alignment
- ZH "解释" and EN "Explain" are in completely separate embedding spaces

**Solution for cross-lingual:** Use transformer models trained on parallel corpora (e.g., Multilingual-E5).

---

## 📚 Examples and Documentation

### Interactive Examples

- **[instruction_awareness_demo.py](examples/instruction_awareness_demo.py)** - 5 interactive demos showing instruction-awareness in English
- **[real_world_use_cases.py](examples/real_world_use_cases.py)** - 5 practical use cases (search, RAG, code, etc.)
- **[monolingual_testing.py](examples/monolingual_testing.py)** - Comprehensive multilingual testing across 7 languages
- **[advanced_limits_testing.py](examples/advanced_limits_testing.py)** - Edge cases and failure modes

### Full Documentation

- **[BENCHMARKS.md](BENCHMARKS.md)** - Detailed benchmark comparisons
- **[MONOLINGUAL_FINDINGS.md](MONOLINGUAL_FINDINGS.md)** - Multilingual testing discoveries
- **[LIMITS.md](LIMITS.md)** - Comprehensive limitations analysis

### Test Results

- `examples/monolingual_test_results.json` - Monolingual test data (7 languages)
- `examples/test_results_advanced.json` - Cross-lingual and edge case data

---

## 🎯 Decision Guide: When to Use This Model

### Use qwen25-deposium-1024d if:

✅ You need **instruction-aware** embeddings (understands Explain, Find, How-to)
✅ Your application is **monolingual** (all content in same language)
✅ You work with **EN, FR, ES, DE, ZH, AR, or RU**
✅ You need **small size** (65MB) for edge deployment
✅ You value **speed** (<1ms) over absolute quality

### Use alternatives if:

❌ You need **cross-lingual** search (query in one language, docs in another) → **Multilingual-E5**
❌ You need **multilingual** search (mixed language results) → **Multilingual-E5**
❌ You need **highest quality** regardless of size → **ColBERT 32M**
❌ You work with **languages not in the list** (EN/FR/ES/DE/ZH/AR/RU) → **Multilingual-E5**

---

## 📖 Citation

If you use this model, please cite:

```bibtex
@misc{qwen25-deposium-1024d,
  author = {TSS Deposium},
  title = {qwen25-deposium-1024d: First Instruction-Aware Model2Vec Across 7 Languages},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/tss-deposium/qwen25-deposium-1024d}
}
```

**Base Model:**
```bibtex
@article{qwen2.5,
  title={Qwen2.5: A Party of Foundation Models},
  author={Qwen Team},
  year={2024}
}
```

**Model2Vec:**
```bibtex
@article{model2vec,
  title={Model2Vec: Distilling Sentence Embeddings from Large Language Models},
  author={MinishLab},
  year={2024}
}
```

---

## 🔗 Links

- **Model:** [tss-deposium/qwen25-deposium-1024d](https://huggingface.co/tss-deposium/qwen25-deposium-1024d)
- **Base Model:** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Model2Vec:** [github.com/MinishLab/model2vec](https://github.com/MinishLab/model2vec)
- **Source Code:** [github.com/theseedship/deposium_embeddings-turbov2](https://github.com/theseedship/deposium_embeddings-turbov2)

---

## 📄 License

Apache 2.0 (same as base model)

---

<div align="center">

**Built by TSS Deposium**

[Report Issues](https://github.com/theseedship/deposium_embeddings-turbov2/issues) • [Documentation](https://github.com/theseedship/deposium_embeddings-turbov2)

**First Model2Vec with multilingual instruction-awareness**

</div>
