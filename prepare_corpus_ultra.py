#!/usr/bin/env python3
"""
Ultra-Enhanced Corpus with Multilingual + Legal Content

Adds to existing enhanced corpus:
- Multilingual legal documents (24 languages via Multi_Legal_Pile)
- English legal corpus (Pile of Law)
- Multilingual Wikipedia/scientific content
- European languages: FR, ES, DE, IT, PT, NL, PL, etc.

Target: ~800k diverse multilingual sentences covering:
- Scientific/Technical (existing 500k)
- Legal/Law (150k multilingual)
- Multilingual General/Scientific (150k)
"""

import logging
import json
from pathlib import Path
from typing import List
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_ultra_corpus():
    """Prepare ultra-enhanced multilingual + legal corpus"""

    logger.info("=" * 80)
    logger.info("🌍 Preparing ULTRA Corpus: Multilingual + Legal")
    logger.info("=" * 80)

    try:
        from datasets import load_dataset
        logger.info("✅ datasets library imported")
    except ImportError:
        logger.error("❌ datasets library not installed")
        return False

    output_dir = Path("./data/model2vec_corpus_ultra")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing enhanced corpus (500k English scientific/technical)
    existing_corpus_path = Path("./data/model2vec_corpus_enhanced/corpus.jsonl")
    corpus: List[str] = []

    if existing_corpus_path.exists():
        logger.info(f"\n📂 Loading existing enhanced corpus...")
        with open(existing_corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                corpus.append(data['text'])
        logger.info(f"✅ Loaded {len(corpus)} English sentences from enhanced corpus")
    else:
        logger.warning("⚠️  No existing enhanced corpus found")
        return False

    logger.info("\n📥 Adding Multilingual + Legal Content...")

    # ==================================================================
    # LEGAL DATASETS
    # ==================================================================

    # 1. Multi_Legal_Pile (24 European languages, contracts + caselaw)
    logger.info("\n⚖️  1️⃣ Loading Multilingual Legal Corpus (Multi_Legal_Pile)")
    logger.info("   Languages: 24 European (FR, ES, DE, IT, PT, NL, PL, etc.)")
    logger.info("   Target: 100k sentences")
    try:
        # Multi_Legal_Pile has multiple configurations for different languages
        # We'll sample from multiple languages
        legal_languages = ['bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et',
                          'fi', 'fr', 'hr', 'hu', 'it', 'lt', 'lv', 'mt',
                          'nl', 'pl', 'pt', 'ro', 'sk', 'sl', 'sv']

        legal_count = 0
        target_per_lang = 100000 // len(legal_languages)  # ~4300 per language

        for lang in legal_languages[:10]:  # Sample first 10 languages for efficiency
            try:
                logger.info(f"   Loading legal texts: {lang.upper()}")
                legal_data = load_dataset(
                    "joelniklaus/Multi_Legal_Pile",
                    lang,
                    split="train",
                    streaming=True,
                    trust_remote_code=True
                )

                lang_count = 0
                for item in legal_data:
                    if legal_count >= 100000 or lang_count >= target_per_lang:
                        break

                    text = item.get('text', '')
                    if text and len(text) > 50:
                        # Split into sentences (simple approach)
                        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
                        for sent in sentences[:3]:  # Max 3 sentences per document
                            if legal_count >= 100000:
                                break
                            corpus.append(sent)
                            legal_count += 1
                            lang_count += 1

                logger.info(f"      ✅ {lang.upper()}: {lang_count} sentences")

            except Exception as e:
                logger.warning(f"      ⚠️  {lang.upper()} failed: {e}")
                continue

        logger.info(f"✅ Multilingual Legal: {legal_count} sentences collected")

    except Exception as e:
        logger.warning(f"⚠️  Multi_Legal_Pile failed: {e}")

    # 2. Pile of Law (English legal corpus)
    logger.info("\n⚖️  2️⃣ Loading English Legal Corpus (Pile of Law)")
    logger.info("   Target: 50k sentences")
    try:
        pile_of_law = load_dataset(
            "pile-of-law/pile-of-law",
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        law_count = 0
        for item in pile_of_law:
            if law_count >= 50000:
                break

            text = item.get('text', '')
            if text and len(text) > 50:
                sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
                corpus.extend(sentences[:2])
                law_count += len(sentences[:2])

        logger.info(f"✅ English Legal: {law_count} sentences collected")

    except Exception as e:
        logger.warning(f"⚠️  Pile of Law failed: {e}")

    # ==================================================================
    # MULTILINGUAL GENERAL/SCIENTIFIC
    # ==================================================================

    # 3. Multilingual Wikipedia
    logger.info("\n🌍 3️⃣ Loading Multilingual Wikipedia")
    logger.info("   Languages: FR, ES, DE, IT, PT, NL")
    logger.info("   Target: 100k sentences")

    wiki_languages = {
        'fr': 20000,  # French
        'es': 20000,  # Spanish
        'de': 20000,  # German
        'it': 15000,  # Italian
        'pt': 15000,  # Portuguese
        'nl': 10000,  # Dutch
    }

    wiki_total = 0
    for lang, target in wiki_languages.items():
        try:
            logger.info(f"   Loading Wikipedia: {lang.upper()}")
            wiki = load_dataset(
                "wikimedia/wikipedia",
                f"20231101.{lang}",
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            lang_count = 0
            for item in wiki:
                if lang_count >= target:
                    break

                text = item.get('text', '')
                if text and len(text) > 50:
                    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
                    corpus.extend(sentences[:5])
                    lang_count += len(sentences[:5])

            wiki_total += lang_count
            logger.info(f"      ✅ {lang.upper()}: {lang_count} sentences")

        except Exception as e:
            logger.warning(f"      ⚠️  {lang.upper()} Wikipedia failed: {e}")

    logger.info(f"✅ Multilingual Wikipedia: {wiki_total} sentences collected")

    # 4. Multilingual Scientific Papers (if available)
    logger.info("\n🔬 4️⃣ Loading Multilingual Scientific Content")
    logger.info("   Target: 50k sentences")
    try:
        # Try multilingual scientific abstracts
        sci_multi = load_dataset(
            "allenai/scirepeval",
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        sci_count = 0
        for item in sci_multi:
            if sci_count >= 50000:
                break

            text = item.get('abstract', '') or item.get('text', '')
            if text and len(text) > 50:
                sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
                corpus.extend(sentences[:3])
                sci_count += len(sentences[:3])

        logger.info(f"✅ Multilingual Science: {sci_count} sentences collected")

    except Exception as e:
        logger.warning(f"⚠️  Multilingual science failed: {e}")

    # ==================================================================
    # CLEAN AND FINALIZE
    # ==================================================================

    logger.info("\n🧹 Cleaning ultra corpus...")

    # Remove duplicates
    corpus_unique = list(set(corpus))
    logger.info(f"  Removed {len(corpus) - len(corpus_unique)} duplicates")

    # Filter by length (20-500 characters)
    corpus_filtered = [s for s in corpus_unique if 20 <= len(s) <= 500]
    logger.info(f"  Filtered by length: {len(corpus_filtered)} sentences remaining")

    # Shuffle
    random.shuffle(corpus_filtered)

    # Limit to 800k (or keep all if less)
    corpus_final = corpus_filtered[:800000]

    logger.info(f"\n📊 Final ultra corpus statistics:")
    logger.info(f"  Total sentences: {len(corpus_final)}")
    logger.info(f"  Avg length: {sum(len(s) for s in corpus_final) / len(corpus_final):.1f} chars")
    logger.info(f"  Min length: {min(len(s) for s in corpus_final)} chars")
    logger.info(f"  Max length: {max(len(s) for s in corpus_final)} chars")

    # Estimated domain/language breakdown
    logger.info(f"\n📈 Estimated distribution:")
    logger.info(f"  English Scientific/Technical: ~500k sentences")
    logger.info(f"  Multilingual Legal: ~150k sentences")
    logger.info(f"  Multilingual General/Scientific: ~150k sentences")
    logger.info(f"  Languages: EN, FR, ES, DE, IT, PT, NL, PL, BG, CS, DA, EL, etc.")

    # Save corpus
    logger.info(f"\n💾 Saving ultra corpus...")

    corpus_file = output_dir / "corpus.jsonl"
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for sentence in corpus_final:
            json.dump({"text": sentence}, f, ensure_ascii=False)
            f.write('\n')

    corpus_txt = output_dir / "corpus_sample.txt"
    with open(corpus_txt, 'w', encoding='utf-8') as f:
        for sentence in corpus_final[:1000]:
            f.write(sentence + '\n')

    corpus_size_mb = corpus_file.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Ultra corpus saved: {corpus_size_mb:.1f} MB")

    logger.info("\n" + "=" * 80)
    logger.info("✅ ULTRA CORPUS PREPARATION COMPLETE")
    logger.info("=" * 80)

    logger.info(f"\n📁 Files created:")
    logger.info(f"  - {corpus_file} ({corpus_size_mb:.1f} MB)")
    logger.info(f"  - {corpus_txt} (sample)")

    logger.info(f"\n🚀 Next step:")
    logger.info(f"  python3 distill_model2vec_optimized.py --corpus-path {corpus_file}")

    return True


def main():
    """Main execution"""
    import sys

    success = prepare_ultra_corpus()

    if success:
        logger.info("\n🎉 Ultra corpus preparation successful!")
        sys.exit(0)
    else:
        logger.error("\n❌ Ultra corpus preparation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
