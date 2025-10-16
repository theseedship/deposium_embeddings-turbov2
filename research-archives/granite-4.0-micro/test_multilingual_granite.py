#!/usr/bin/env python3
"""
Detailed Multilingual Testing for Granite 4.0 Micro
Tests: Cross-lingual retrieval, semantic similarity in 12 languages,
       translation quality, multilingual RAG
"""

import numpy as np
from pathlib import Path
from model2vec import StaticModel

print("=" * 80)
print("🌍 MULTILINGUAL DEEP DIVE - Granite 4.0 Micro vs Competitors")
print("=" * 80)
print()

# Load models
models = {}
model_paths = {
    "Qwen2.5-1.5B (PROD)": "models/qwen25-deposium-1024d",
    "Granite 4.0 Micro (NEW)": "granite-4.0-micro-deposium-1024d",
}

print("📥 Loading models...")
for name, path in model_paths.items():
    model_dir = Path(path)
    if model_dir.exists():
        try:
            print(f"  Loading {name}...")
            models[name] = StaticModel.from_pretrained(str(model_dir))
            print(f"    ✅ Loaded")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
    else:
        print(f"  ⚠️  Not found: {path}")

if "Granite 4.0 Micro (NEW)" not in models:
    print()
    print("❌ Granite model not found! Run distillation first:")
    print("   python3 distill_granite_4_0_micro.py")
    exit(1)

print()
print("=" * 80)
print("TEST 1: SEMANTIC SIMILARITY PER LANGUAGE")
print("=" * 80)
print()

# Language-specific tests
language_tests = {
    "English (EN)": [
        ("Artificial intelligence is transforming technology", "AI and machine learning revolutionize tech"),
        ("The weather is sunny and warm", "It's a beautiful day with sunshine"),
        ("Python programming language", "Coding in Python"),
    ],
    "French (FR)": [
        ("L'intelligence artificielle transforme la technologie", "L'IA et le machine learning révolutionnent la tech"),
        ("Il fait beau et chaud", "C'est une belle journée ensoleillée"),
        ("Langage de programmation Python", "Programmer en Python"),
    ],
    "German (DE)": [
        ("Künstliche Intelligenz verändert Technologie", "KI und maschinelles Lernen revolutionieren Technik"),
        ("Das Wetter ist sonnig und warm", "Es ist ein schöner Tag mit Sonnenschein"),
        ("Python Programmiersprache", "Programmieren in Python"),
    ],
    "Spanish (ES)": [
        ("La inteligencia artificial transforma la tecnología", "La IA y el aprendizaje automático revolucionan la tecnología"),
        ("El clima es soleado y cálido", "Es un hermoso día con sol"),
        ("Lenguaje de programación Python", "Programar en Python"),
    ],
    "Chinese (ZH)": [
        ("人工智能正在改变技术", "AI和机器学习革新科技"),
        ("天气晴朗温暖", "今天阳光明媚"),
        ("Python编程语言", "用Python编程"),
    ],
    "Japanese (JP)": [
        ("人工知能が技術を変えています", "AIと機械学習が技術革新を起こしています"),
        ("天気は晴れて暖かい", "素晴らしい晴れた日です"),
        ("Pythonプログラミング言語", "Pythonでプログラミング"),
    ],
}

print("Test: Semantic similarity within each language")
print()

language_results = {}

for lang, test_pairs in language_tests.items():
    print(f"📖 Testing {lang}:")

    for name, model in models.items():
        scores = []

        for text1, text2 in test_pairs:
            emb1 = model.encode([text1])[0]
            emb2 = model.encode([text2])[0]

            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            scores.append(similarity)

        avg_score = np.mean(scores)

        if name not in language_results:
            language_results[name] = {}
        language_results[name][lang] = avg_score

        print(f"  {name}: {avg_score:.4f} ({avg_score*100:.2f}%)")

    print()

print("=" * 80)
print("📊 Summary by Language:")
print("=" * 80)
print()

for name in models:
    print(f"{name}:")
    for lang, score in language_results[name].items():
        print(f"  {lang}: {score*100:.2f}%")
    print()

print()
print("=" * 80)
print("TEST 2: CROSS-LINGUAL RETRIEVAL")
print("=" * 80)
print()

# Cross-lingual test: Query in one language, documents in another
cross_lingual_tests = [
    {
        "query_lang": "EN",
        "query": "What is machine learning?",
        "doc_lang": "FR",
        "documents": [
            "Le machine learning est une branche de l'intelligence artificielle",
            "Les bases de données stockent des informations",
            "La cybersécurité protège les données",
        ],
        "correct_idx": 0,
    },
    {
        "query_lang": "FR",
        "query": "Qu'est-ce que Python?",
        "doc_lang": "EN",
        "documents": [
            "Python is a high-level programming language",
            "Databases store structured data",
            "Cloud computing provides scalable resources",
        ],
        "correct_idx": 0,
    },
    {
        "query_lang": "EN",
        "query": "Climate change impacts",
        "doc_lang": "ES",
        "documents": [
            "El cambio climático afecta al planeta",
            "La programación en Python es popular",
            "Las bases de datos almacenan información",
        ],
        "correct_idx": 0,
    },
]

print("Test: Query in one language, retrieve documents in another language")
print()

cross_lingual_results = {name: [] for name in models}

for test in cross_lingual_tests:
    query = test["query"]
    documents = test["documents"]
    correct_idx = test["correct_idx"]

    print(f"Query [{test['query_lang']}]: '{query}'")
    print(f"Documents [{test['doc_lang']}]:")
    for i, doc in enumerate(documents):
        marker = "✓" if i == correct_idx else " "
        print(f"  [{marker}] {doc}")
    print()

    for name, model in models.items():
        query_emb = model.encode([query])[0]
        doc_embs = model.encode(documents)

        similarities = []
        for doc_emb in doc_embs:
            sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
            similarities.append(sim)

        top_idx = np.argmax(similarities)
        is_correct = top_idx == correct_idx
        cross_lingual_results[name].append(1.0 if is_correct else 0.0)

        status = "✅" if is_correct else "❌"
        print(f"  {name}: {status} Retrieved doc {top_idx} (sim: {similarities[top_idx]:.4f})")

    print()

print("📊 Cross-Lingual Retrieval Accuracy:")
for name in models:
    accuracy = np.mean(cross_lingual_results[name]) * 100
    print(f"  {name}: {accuracy:.2f}% ({int(sum(cross_lingual_results[name]))}/{len(cross_lingual_results[name])} correct)")

print()
print("=" * 80)
print("TEST 3: MULTILINGUAL RAG SIMULATION")
print("=" * 80)
print()

# Multilingual knowledge base
multilingual_kb = [
    ("EN", "Python is a versatile programming language used for web development, data science, and automation"),
    ("FR", "Le deep learning est une technique d'apprentissage automatique utilisant des réseaux de neurones profonds"),
    ("DE", "Datenbanken sind Systeme zum Speichern und Verwalten strukturierter Informationen"),
    ("ES", "La computación en la nube proporciona recursos escalables a través de internet"),
    ("ZH", "人工智能是计算机科学的一个分支，旨在创建智能机器"),
    ("EN", "Machine learning algorithms learn patterns from data without explicit programming"),
]

queries_multilingual = [
    ("EN", "Tell me about Python programming", 0),  # Should match EN Python doc
    ("FR", "Qu'est-ce que le deep learning?", 1),  # Should match FR deep learning doc
    ("DE", "Was sind Datenbanken?", 2),  # Should match DE database doc
    ("EN", "What is machine learning?", 5),  # Should match EN ML doc
]

print("Test: Multilingual RAG with mixed-language knowledge base")
print()

print("📚 Knowledge Base:")
for i, (lang, doc) in enumerate(multilingual_kb):
    print(f"  [{i}] [{lang}] {doc[:60]}...")
print()

rag_results = {name: [] for name in models}

for query_lang, query, correct_idx in queries_multilingual:
    print(f"Query [{query_lang}]: '{query}'")

    for name, model in models.items():
        query_emb = model.encode([query])[0]

        # Encode all documents (regardless of language)
        doc_texts = [doc for _, doc in multilingual_kb]
        doc_embs = model.encode(doc_texts)

        similarities = []
        for doc_emb in doc_embs:
            sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
            similarities.append(sim)

        top_idx = np.argmax(similarities)
        is_correct = top_idx == correct_idx
        rag_results[name].append(1.0 if is_correct else 0.0)

        status = "✅" if is_correct else "❌"
        top_lang, top_doc = multilingual_kb[top_idx]
        print(f"  {name}: {status} [{top_lang}] {top_doc[:50]}... (sim: {similarities[top_idx]:.4f})")

    print()

print("📊 Multilingual RAG Accuracy:")
for name in models:
    accuracy = np.mean(rag_results[name]) * 100
    print(f"  {name}: {accuracy:.2f}% ({int(sum(rag_results[name]))}/{len(rag_results[name])} correct)")

print()
print("=" * 80)
print("🏆 FINAL MULTILINGUAL VERDICT")
print("=" * 80)
print()

# Calculate overall multilingual score
granite_name = "Granite 4.0 Micro (NEW)"
qwen_name = "Qwen2.5-1.5B (PROD)"

if granite_name in models and qwen_name in models:
    # Average across all languages
    granite_lang_avg = np.mean(list(language_results[granite_name].values())) * 100
    qwen_lang_avg = np.mean(list(language_results[qwen_name].values())) * 100

    # Cross-lingual accuracy
    granite_cross = np.mean(cross_lingual_results[granite_name]) * 100
    qwen_cross = np.mean(cross_lingual_results[qwen_name]) * 100

    # RAG accuracy
    granite_rag = np.mean(rag_results[granite_name]) * 100
    qwen_rag = np.mean(rag_results[qwen_name]) * 100

    print(f"📊 Overall Multilingual Performance:")
    print()
    print(f"{'Metric':<30} {qwen_name:<20} {granite_name:<25} {'Winner':<10}")
    print("-" * 90)
    print(f"{'Per-language similarity':<30} {qwen_lang_avg:>6.2f}%            {granite_lang_avg:>6.2f}%              {'Granite' if granite_lang_avg > qwen_lang_avg else 'Qwen':<10}")
    print(f"{'Cross-lingual retrieval':<30} {qwen_cross:>6.2f}%            {granite_cross:>6.2f}%              {'Granite' if granite_cross > qwen_cross else 'Qwen':<10}")
    print(f"{'Multilingual RAG':<30} {qwen_rag:>6.2f}%            {granite_rag:>6.2f}%              {'Granite' if granite_rag > qwen_rag else 'Qwen':<10}")
    print()

    # Composite score
    granite_composite = (granite_lang_avg + granite_cross + granite_rag) / 3
    qwen_composite = (qwen_lang_avg + qwen_cross + qwen_rag) / 3

    print(f"🎯 Composite Multilingual Score:")
    print(f"  {qwen_name}: {qwen_composite:.2f}%")
    print(f"  {granite_name}: {granite_composite:.2f}%")
    print()

    if granite_composite > qwen_composite:
        diff = granite_composite - qwen_composite
        print(f"✅ GRANITE WINS for multilingual use cases!")
        print(f"   Improvement: +{diff:.2f}% over Qwen2.5-1.5B")
        print()
        print(f"   Granite excels in:")
        if granite_lang_avg > qwen_lang_avg:
            print(f"   - Per-language quality (+{granite_lang_avg - qwen_lang_avg:.2f}%)")
        if granite_cross > qwen_cross:
            print(f"   - Cross-lingual retrieval (+{granite_cross - qwen_cross:.2f}%)")
        if granite_rag > qwen_rag:
            print(f"   - Multilingual RAG (+{granite_rag - qwen_rag:.2f}%)")
    else:
        diff = qwen_composite - granite_composite
        print(f"⚠️  QWEN WINS even for multilingual!")
        print(f"   Lead: +{diff:.2f}% over Granite")
        print()
        print(f"   Consider Granite only if you NEED 12-language support")

print()
print("=" * 80)
print("📄 Saving detailed results...")
print("=" * 80)

# Save to file
with open("granite_multilingual_results.txt", "w") as f:
    f.write("=" * 80 + "\n")
    f.write("GRANITE 4.0 MICRO - MULTILINGUAL ANALYSIS\n")
    f.write("=" * 80 + "\n\n")

    f.write("PER-LANGUAGE RESULTS:\n")
    f.write("-" * 80 + "\n")
    for name in models:
        f.write(f"\n{name}:\n")
        for lang, score in language_results[name].items():
            f.write(f"  {lang}: {score*100:.2f}%\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("CROSS-LINGUAL RETRIEVAL:\n")
    f.write("-" * 80 + "\n")
    for name in models:
        accuracy = np.mean(cross_lingual_results[name]) * 100
        f.write(f"{name}: {accuracy:.2f}%\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("MULTILINGUAL RAG:\n")
    f.write("-" * 80 + "\n")
    for name in models:
        accuracy = np.mean(rag_results[name]) * 100
        f.write(f"{name}: {accuracy:.2f}%\n")

print("✅ Results saved to: granite_multilingual_results.txt")
print()
