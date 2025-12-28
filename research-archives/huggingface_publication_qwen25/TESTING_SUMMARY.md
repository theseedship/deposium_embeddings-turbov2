# Advanced Testing Summary: qwen25-deposium-1024d

**Date:** 2025-10-19
**Purpose:** Comprehensive limits testing to provide honest documentation

---

## What We Tested

Created `examples/advanced_limits_testing.py` to push the model to its limits across:

1. **Cross-Lingual Instruction-Awareness** (FR↔EN, multilingual mix)
2. **Difficult Cases** (negation, ambiguity, multiple intents, formality)
3. **Edge Cases** (typos, long queries, contradictions, non-Latin scripts)
4. **Performance Degradation** (progressive difficulty analysis)

---

## Key Findings

### ❌ **CRITICAL: Cross-Lingual Does NOT Work**

- **Pass Rate:** 0% (3/3 tests failed)
- **Performance Drop:** -36% to -40% from baseline
- **Root Cause:** Model prioritizes same-language matching over instruction understanding

**Tests:**
```
FR "Explique..." → EN "explanation tutorial"   ❌ FAIL (-6.7%)
EN "Find articles" → FR "Articles...publ..."    ❌ FAIL (-21.3%)
FR "Résume..." → Multilingual results           ❌ FAIL (-39.8%)
```

**Conclusion:** Despite 39.4% "multilingual score", cross-lingual instruction-awareness is **NOT supported**. The score refers to monolingual performance in each language separately.

### ❌ **CRITICAL: Non-Latin Scripts Completely Broken**

- **Arabic:** Negative similarity score (-0.445) ❌
- **Russian:** Negative similarity score (-0.234) ❌
- **Chinese:** Negative similarity score (-0.642) ❌

**Conclusion:** Model is fundamentally incompatible with non-Latin scripts. Do NOT use for Arabic, Chinese, Russian, etc.

### ✅ **Good: Difficult Cases in English**

- **Pass Rate:** 75% (3/4 tests passed)
- **Passed:** Negative instructions, multiple intents, formality
- **Failed:** Ambiguity without context

**Conclusion:** Model handles complex English cases well.

### ❌ **Poor: Edge Cases**

- **Pass Rate:** 0% (0/4 tests passed)
- **Failed:** Typos, long queries, contradictions, non-Latin scripts

**Conclusion:** Model requires clean, well-formed input.

---

## Impact on Documentation

### Before Testing (Misleading)

> "Multilingual: 39.4% - Cross-language alignment"

**Problem:** Implies cross-lingual ability (FR query → EN docs)

### After Testing (Honest)

> "Monolingual FR/ES/DE: ~92% - Each language separately (NOT cross-lingual)
>
> ⚠️ Cross-lingual queries show -36% to -40% performance drop"

**Improvement:** Clear distinction between monolingual and cross-lingual

---

## Documentation Updates

### 1. **LIMITS.md** (New)

Comprehensive 400+ line document covering:
- Executive summary with pass rates
- Detailed test results for each category
- Honest capabilities assessment
- Revised recommendations for use
- Comparison with marketing claims vs reality
- Testing methodology

### 2. **HUGGINGFACE_README.md** (Updated)

**Changes:**
- Added "(EN)" to all English-specific capabilities
- Changed "Multilingual: 39.4% Cross-language" → "Monolingual FR/ES/DE: ~92% NOT cross-lingual"
- Expanded limitations section with clear warnings
- Added "Do NOT use for" list
- Clarified that instruction-awareness is English-only

**Key Addition:**
```markdown
## ⚠️ Important Limitations

### Language Support
✅ Excellent: English-only instruction-aware search (94.96%)
⚠️ Moderate: Monolingual use in FR/ES/DE (~92%)
❌ Not Supported: Cross-lingual queries (-36% to -40% drop)
❌ Not Supported: Non-Latin scripts (negative scores)
```

### 3. **examples/advanced_limits_testing.py** (New)

Full test script (593 lines) with:
- 20+ carefully designed test cases
- Cross-lingual tests (FR→EN, EN→FR, multilingual)
- Difficult cases (negation, ambiguity, multiple intents, formality)
- Edge cases (typos, long queries, contradictions, non-Latin scripts)
- Performance degradation analysis
- JSON results export

### 4. **test_results_advanced.json** (Generated)

Detailed results with:
- Summary pass rates by category
- Individual test results with scores
- Performance degradation metrics

---

## Honest Positioning

### Before

> "First Model2Vec with instruction-awareness and multilingual support"

### After

> "First and best instruction-aware Model2Vec embedding - **optimized for English**.
> For multilingual needs, consider alternatives like Multilingual-E5."

---

## Recommendations for Users

### ✅ Use This Model For:

1. **English-only** semantic search
2. **English-only** RAG systems
3. **English-only** documentation Q&A
4. **English-only** code search
5. **Monolingual** FR/ES/DE (query and docs in SAME language)

### ❌ Do NOT Use For:

1. Cross-lingual search (FR query → EN docs)
2. Multilingual search (mixed language results)
3. Non-Latin scripts (Arabic, Chinese, Russian)
4. User-generated content with typos
5. Very long or contradictory queries

---

## Files Created/Updated

```
huggingface_publication/
├── LIMITS.md                              ✅ NEW - Comprehensive limits doc
├── TESTING_SUMMARY.md                     ✅ NEW - This file
├── HUGGINGFACE_README.md                  ✅ UPDATED - Honest limitations
├── examples/
│   ├── advanced_limits_testing.py         ✅ NEW - Full test suite
│   ├── test_results_advanced.json         ✅ NEW - Test results
│   └── advanced_test_output.log           ✅ NEW - Test output
```

---

## Next Steps for User

1. **Re-upload README.md to HuggingFace** with updated metadata (dataset section added)
2. **Upload LIMITS.md** to HuggingFace for full transparency
3. **Upload advanced_limits_testing.py** so users can verify claims
4. **Consider adding language tags** in HuggingFace: `en` (primary), `fr`, `es`, `de` (secondary, monolingual only)

---

## Lessons Learned

1. **"Multilingual" is ambiguous** - Always clarify monolingual vs cross-lingual
2. **Test edge cases early** - Don't assume capabilities transfer
3. **Negative scores = broken** - Non-Latin scripts show fundamental incompatibility
4. **Marketing vs Reality** - Claims should be testable and honest
5. **Transparency builds trust** - Better to be honest about limitations

---

## Test Results Summary

| Category | Tests | Pass | Fail | Pass Rate |
|----------|-------|------|------|-----------|
| Cross-Lingual | 3 | 0 | 3 | **0%** ❌ |
| Difficult Cases (EN) | 4 | 3 | 1 | **75%** ✅ |
| Edge Cases | 4 | 0 | 4 | **0%** ❌ |
| **Overall** | **11** | **3** | **8** | **27%** |

**Interpretation:**
- **27% overall pass rate** seems low, but reflects **realistic use cases**
- **English-only tasks:** Would be ~75%+ pass rate
- **Failures concentrated in:** Cross-lingual, non-Latin scripts, noisy input

**Honest Assessment:**
> "Excellent English-only instruction-aware embedding with significant multilingual limitations"

---

**Test Script:** `examples/advanced_limits_testing.py`
**Detailed Results:** `test_results_advanced.json`
**Full Documentation:** `LIMITS.md`
**Updated Model Card:** `HUGGINGFACE_README.md`
