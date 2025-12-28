#!/usr/bin/env python3
"""
Monolingual Instruction-Awareness Testing: qwen25-deposium-1024d

Test if instruction-awareness works when EVERYTHING is in the SAME language:
- FR query → FR documents
- ES query → ES documents
- DE query → DE documents
- ZH query → ZH documents
- AR query → AR documents
- RU query → RU documents

This is different from cross-lingual testing (FR query → EN docs).
"""

from model2vec import StaticModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def print_header(text, level=1):
    """Print formatted header"""
    if level == 1:
        print("\n" + "=" * 80)
        print(f"  {text}")
        print("=" * 80)
    else:
        print(f"\n{'─' * 80}")
        print(f"  {text}")
        print('─' * 80)


def test_instruction_awareness(model, language, query, docs, expected_rank=0):
    """
    Test instruction-awareness within a single language
    Returns (success, top_idx, scores)
    """
    print(f"\n📝 Query ({language}): \"{query}\"")
    print(f"\n📄 Documents ({language}):")

    query_emb = model.encode([query])[0]
    doc_embs = model.encode(docs)

    similarities = cosine_similarity([query_emb], doc_embs)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    for i, idx in enumerate(sorted_indices, 1):
        score = similarities[idx]
        doc = docs[idx]

        # Check if this is expected top result
        if idx == expected_rank:
            emoji = "✅" if i == 1 else "❌"
        else:
            emoji = "⚪"

        print(f"  {i}. {emoji} [{score:.3f}] {doc}")

    success = sorted_indices[0] == expected_rank
    top_score = similarities[sorted_indices[0]]
    expected_score = similarities[expected_rank]

    return success, sorted_indices[0], similarities, top_score, expected_score


def main():
    print_header("🌍 MONOLINGUAL INSTRUCTION-AWARENESS TESTING")

    print("\n🔄 Loading model...")
    model = StaticModel.from_pretrained("tss-deposium/qwen25-deposium-1024d")
    print("✅ Model loaded!\n")

    results = {}

    # ========================================================================
    # Test 1: French Monolingual (FR → FR)
    # ========================================================================
    print_header("Test 1: FRANÇAIS (FR → FR)", level=1)

    print_header("Test 1.1: 'Explique' instruction en français", level=2)

    success, top_idx, scores, top_score, expected = test_instruction_awareness(
        model,
        language="FR",
        query="Explique comment fonctionnent les réseaux de neurones",
        docs=[
            "Explication détaillée des réseaux de neurones avec tutoriel complet",  # Should match
            "Les réseaux de neurones ont été inventés en 1950",                      # Historical, not explanation
            "Installation de TensorFlow pour réseaux de neurones",                   # Installation, not explanation
        ],
        expected_rank=0
    )

    results['fr_explique'] = {'success': success, 'top_score': top_score, 'expected': expected}
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: FR 'Explique' → explication/tutoriel")
    print(f"   Score: {expected:.3f}")

    print_header("Test 1.2: 'Trouve' instruction en français", level=2)

    success, top_idx, scores, top_score, expected = test_instruction_awareness(
        model,
        language="FR",
        query="Trouve des articles sur le changement climatique",
        docs=[
            "Articles scientifiques et publications sur le changement climatique",  # Articles/publications
            "Le changement climatique est un problème sérieux",                      # Statement, not articles
            "Comment réduire le changement climatique",                              # How-to, not articles
        ],
        expected_rank=0
    )

    results['fr_trouve'] = {'success': success, 'top_score': top_score, 'expected': expected}
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: FR 'Trouve' → articles/publications")
    print(f"   Score: {expected:.3f}")

    # ========================================================================
    # Test 2: Spanish Monolingual (ES → ES)
    # ========================================================================
    print_header("Test 2: ESPAÑOL (ES → ES)", level=1)

    print_header("Test 2.1: 'Explica' instruction en español", level=2)

    success, top_idx, scores, top_score, expected = test_instruction_awareness(
        model,
        language="ES",
        query="Explica cómo funcionan las redes neuronales",
        docs=[
            "Explicación completa de redes neuronales con tutorial detallado",  # Explanation/tutorial
            "Las redes neuronales se utilizan en IA",                            # General statement
            "Instalación de frameworks de redes neuronales",                     # Installation
        ],
        expected_rank=0
    )

    results['es_explica'] = {'success': success, 'top_score': top_score, 'expected': expected}
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: ES 'Explica' → explicación/tutorial")
    print(f"   Score: {expected:.3f}")

    print_header("Test 2.2: 'Encuentra' instruction en español", level=2)

    success, top_idx, scores, top_score, expected = test_instruction_awareness(
        model,
        language="ES",
        query="Encuentra artículos sobre cambio climático",
        docs=[
            "Artículos científicos y publicaciones sobre cambio climático",  # Articles/publications
            "El cambio climático es un problema global",                      # Statement
            "Cómo combatir el cambio climático",                              # How-to
        ],
        expected_rank=0
    )

    results['es_encuentra'] = {'success': success, 'top_score': top_score, 'expected': expected}
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: ES 'Encuentra' → artículos/publicaciones")
    print(f"   Score: {expected:.3f}")

    # ========================================================================
    # Test 3: German Monolingual (DE → DE)
    # ========================================================================
    print_header("Test 3: DEUTSCH (DE → DE)", level=1)

    print_header("Test 3.1: 'Erkläre' instruction en allemand", level=2)

    success, top_idx, scores, top_score, expected = test_instruction_awareness(
        model,
        language="DE",
        query="Erkläre wie neuronale Netze funktionieren",
        docs=[
            "Ausführliche Erklärung neuronaler Netze mit Tutorial",  # Explanation/tutorial
            "Neuronale Netze werden in KI verwendet",                 # General statement
            "Installation von neuronalen Netz-Frameworks",            # Installation
        ],
        expected_rank=0
    )

    results['de_erklaere'] = {'success': success, 'top_score': top_score, 'expected': expected}
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: DE 'Erkläre' → Erklärung/Tutorial")
    print(f"   Score: {expected:.3f}")

    print_header("Test 3.2: 'Finde' instruction en allemand", level=2)

    success, top_idx, scores, top_score, expected = test_instruction_awareness(
        model,
        language="DE",
        query="Finde Artikel über Klimawandel",
        docs=[
            "Wissenschaftliche Artikel und Publikationen über Klimawandel",  # Articles/publications
            "Klimawandel ist ein ernstes Problem",                            # Statement
            "Wie man den Klimawandel bekämpft",                               # How-to
        ],
        expected_rank=0
    )

    results['de_finde'] = {'success': success, 'top_score': top_score, 'expected': expected}
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: DE 'Finde' → Artikel/Publikationen")
    print(f"   Score: {expected:.3f}")

    # ========================================================================
    # Test 4: Chinese Monolingual (ZH → ZH)
    # ========================================================================
    print_header("Test 4: 中文 (ZH → ZH)", level=1)

    print_header("Test 4.1: '解释' instruction en chinois", level=2)

    success, top_idx, scores, top_score, expected = test_instruction_awareness(
        model,
        language="ZH",
        query="解释神经网络如何工作",
        docs=[
            "神经网络详细解释和教程指南",  # Explanation/tutorial
            "神经网络在人工智能中使用",    # General statement
            "安装神经网络框架",            # Installation
        ],
        expected_rank=0
    )

    results['zh_jieshi'] = {'success': success, 'top_score': top_score, 'expected': expected}
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: ZH '解释' → 解释/教程")
    print(f"   Score: {expected:.3f}")

    print_header("Test 4.2: '查找' instruction en chinois", level=2)

    success, top_idx, scores, top_score, expected = test_instruction_awareness(
        model,
        language="ZH",
        query="查找关于气候变化的文章",
        docs=[
            "气候变化科学文章和出版物",  # Articles/publications
            "气候变化是一个严重问题",    # Statement
            "如何应对气候变化",          # How-to
        ],
        expected_rank=0
    )

    results['zh_chazhao'] = {'success': success, 'top_score': top_score, 'expected': expected}
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: ZH '查找' → 文章/出版物")
    print(f"   Score: {expected:.3f}")

    # ========================================================================
    # Test 5: Arabic Monolingual (AR → AR)
    # ========================================================================
    print_header("Test 5: العربية (AR → AR)", level=1)

    print_header("Test 5.1: 'اشرح' instruction en arabe", level=2)

    success, top_idx, scores, top_score, expected = test_instruction_awareness(
        model,
        language="AR",
        query="اشرح كيف تعمل الشبكات العصبية",
        docs=[
            "شرح مفصل للشبكات العصبية مع دليل تعليمي",  # Explanation/tutorial
            "الشبكات العصبية تستخدم في الذكاء الاصطناعي",  # General statement
            "تثبيت أطر الشبكات العصبية",                   # Installation
        ],
        expected_rank=0
    )

    results['ar_ishrah'] = {'success': success, 'top_score': top_score, 'expected': expected}
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: AR 'اشرح' → شرح/دليل")
    print(f"   Score: {expected:.3f}")

    print_header("Test 5.2: 'ابحث' instruction en arabe", level=2)

    success, top_idx, scores, top_score, expected = test_instruction_awareness(
        model,
        language="AR",
        query="ابحث عن مقالات حول تغير المناخ",
        docs=[
            "مقالات علمية ومنشورات حول تغير المناخ",  # Articles/publications
            "تغير المناخ مشكلة خطيرة",                 # Statement
            "كيفية مكافحة تغير المناخ",                # How-to
        ],
        expected_rank=0
    )

    results['ar_ibhath'] = {'success': success, 'top_score': top_score, 'expected': expected}
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: AR 'ابحث' → مقالات/منشورات")
    print(f"   Score: {expected:.3f}")

    # ========================================================================
    # Test 6: Russian Monolingual (RU → RU)
    # ========================================================================
    print_header("Test 6: РУССКИЙ (RU → RU)", level=1)

    print_header("Test 6.1: 'Объясни' instruction en russe", level=2)

    success, top_idx, scores, top_score, expected = test_instruction_awareness(
        model,
        language="RU",
        query="Объясни как работают нейронные сети",
        docs=[
            "Подробное объяснение нейронных сетей с учебным пособием",  # Explanation/tutorial
            "Нейронные сети используются в ИИ",                          # General statement
            "Установка фреймворков нейронных сетей",                     # Installation
        ],
        expected_rank=0
    )

    results['ru_obyasni'] = {'success': success, 'top_score': top_score, 'expected': expected}
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: RU 'Объясни' → объяснение/пособие")
    print(f"   Score: {expected:.3f}")

    print_header("Test 6.2: 'Найди' instruction en russe", level=2)

    success, top_idx, scores, top_score, expected = test_instruction_awareness(
        model,
        language="RU",
        query="Найди статьи о изменении климата",
        docs=[
            "Научные статьи и публикации об изменении климата",  # Articles/publications
            "Изменение климата это серьезная проблема",           # Statement
            "Как бороться с изменением климата",                  # How-to
        ],
        expected_rank=0
    )

    results['ru_naidi'] = {'success': success, 'top_score': top_score, 'expected': expected}
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: RU 'Найди' → статьи/публикации")
    print(f"   Score: {expected:.3f}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print_header("📊 MONOLINGUAL INSTRUCTION-AWARENESS SUMMARY", level=1)

    # Calculate pass rates by language
    languages = {
        'Français (FR)': ['fr_explique', 'fr_trouve'],
        'Español (ES)': ['es_explica', 'es_encuentra'],
        'Deutsch (DE)': ['de_erklaere', 'de_finde'],
        '中文 (ZH)': ['zh_jieshi', 'zh_chazhao'],
        'العربية (AR)': ['ar_ishrah', 'ar_ibhath'],
        'Русский (RU)': ['ru_obyasni', 'ru_naidi'],
    }

    print("\n╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                     MONOLINGUAL TEST RESULTS                                  ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝\n")

    overall_pass = 0
    overall_total = 0

    for lang_name, test_keys in languages.items():
        pass_count = sum(1 for key in test_keys if results[key]['success'])
        total_count = len(test_keys)
        pass_rate = (pass_count / total_count) * 100

        overall_pass += pass_count
        overall_total += total_count

        # Get average score
        avg_score = np.mean([results[key]['expected'] for key in test_keys])

        emoji = "✅" if pass_rate >= 50 else "⚠️" if pass_rate > 0 else "❌"

        print(f"{emoji} {lang_name:20s}: {pass_count}/{total_count} tests passed ({pass_rate:.0f}%)")
        print(f"   Average score: {avg_score:.3f}")

    overall_rate = (overall_pass / overall_total) * 100

    print(f"\n{'=' * 80}")
    print(f"OVERALL: {overall_pass}/{overall_total} tests passed ({overall_rate:.0f}%)")
    print(f"{'=' * 80}\n")

    # Analysis
    print("🔬 ANALYSIS:\n")

    # Group by script type
    latin_tests = ['fr_explique', 'fr_trouve', 'es_explica', 'es_encuentra', 'de_erklaere', 'de_finde']
    non_latin_tests = ['zh_jieshi', 'zh_chazhao', 'ar_ishrah', 'ar_ibhath', 'ru_obyasni', 'ru_naidi']

    latin_pass = sum(1 for key in latin_tests if results[key]['success'])
    latin_total = len(latin_tests)
    latin_rate = (latin_pass / latin_total) * 100

    non_latin_pass = sum(1 for key in non_latin_tests if results[key]['success'])
    non_latin_total = len(non_latin_tests)
    non_latin_rate = (non_latin_pass / non_latin_total) * 100

    latin_avg_score = np.mean([results[key]['expected'] for key in latin_tests])
    non_latin_avg_score = np.mean([results[key]['expected'] for key in non_latin_tests])

    print(f"📊 Latin Scripts (FR/ES/DE):")
    print(f"   Pass rate: {latin_rate:.0f}% ({latin_pass}/{latin_total})")
    print(f"   Average score: {latin_avg_score:.3f}")

    print(f"\n📊 Non-Latin Scripts (ZH/AR/RU):")
    print(f"   Pass rate: {non_latin_rate:.0f}% ({non_latin_pass}/{non_latin_total})")
    print(f"   Average score: {non_latin_avg_score:.3f}")

    # Conclusion
    print(f"\n💡 CONCLUSIONS:\n")

    if latin_rate > 50:
        print("✅ Latin-script languages (FR/ES/DE): Instruction-awareness WORKS monolingual")
    else:
        print("❌ Latin-script languages (FR/ES/DE): Instruction-awareness DOES NOT WORK")

    if non_latin_rate > 50:
        print("✅ Non-Latin scripts (ZH/AR/RU): Instruction-awareness WORKS monolingual")
    else:
        print("❌ Non-Latin scripts (ZH/AR/RU): Instruction-awareness DOES NOT WORK")

    # Compare with EN baseline (94.96%)
    en_baseline = 0.9496
    print(f"\n📉 Performance vs English Baseline (94.96%):")
    print(f"   Latin scripts: -{(en_baseline - latin_avg_score)*100:.1f}% ({latin_avg_score:.1%} vs {en_baseline:.1%})")
    print(f"   Non-Latin scripts: -{(en_baseline - non_latin_avg_score)*100:.1f}% ({non_latin_avg_score:.1%} vs {en_baseline:.1%})")

    # Save results
    print("\n💾 Saving results to monolingual_test_results.json...")
    import json

    output = {
        'summary': {
            'overall_pass_rate': overall_rate / 100,
            'latin_scripts_pass_rate': latin_rate / 100,
            'non_latin_scripts_pass_rate': non_latin_rate / 100,
            'latin_avg_score': float(latin_avg_score),
            'non_latin_avg_score': float(non_latin_avg_score)
        },
        'by_language': {
            lang_name: {
                'tests': {
                    key: {
                        'success': bool(results[key]['success']),
                        'score': float(results[key]['expected'])
                    }
                    for key in test_keys
                },
                'pass_rate': float(sum(1 for key in test_keys if results[key]['success']) / len(test_keys))
            }
            for lang_name, test_keys in languages.items()
        },
        'all_results': {
            key: {
                'success': bool(value['success']),
                'score': float(value['expected'])
            }
            for key, value in results.items()
        }
    }

    with open('monolingual_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("✅ Results saved!")

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      RECOMMENDATION UPDATE                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Based on these results, the model's monolingual instruction-awareness is:

✅ GOOD for: Latin scripts (FR/ES/DE) monolingual use - {latin_rate:.0f}% pass rate
❌ POOR for: Non-Latin scripts (ZH/AR/RU) monolingual use - {non_latin_rate:.0f}% pass rate

This confirms: The model is optimized for English and other Latin-script
languages, but NOT for non-Latin scripts even in monolingual mode.
""")


if __name__ == "__main__":
    main()
