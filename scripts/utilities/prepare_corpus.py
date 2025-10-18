#!/usr/bin/env python3
"""
Prepare Diverse Corpus for Model2Vec Distillation

Downloads and prepares a high-quality, diverse corpus from multiple sources:
- Wikipedia (general knowledge)
- Academic papers (technical content)
- Q&A forums (conversational)
- Web content (diverse domains)

Total target: ~500k sentences for efficient distillation
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


def prepare_corpus():
    """Prepare diverse corpus for Model2Vec distillation"""

    logger.info("=" * 80)
    logger.info("📚 Preparing Diverse Corpus for Model2Vec Distillation")
    logger.info("=" * 80)

    # Check dependencies
    try:
        from datasets import load_dataset
        logger.info("✅ datasets library imported")
    except ImportError:
        logger.error("❌ datasets library not installed")
        logger.error("Install: pip install datasets")
        return False

    output_dir = Path("./data/model2vec_corpus")
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus: List[str] = []

    logger.info("\n📥 Downloading datasets...")
    logger.info("This may take 10-20 minutes on first run (datasets are cached)")

    # 1. Wikipedia (30% - general knowledge)
    logger.info("\n1️⃣ Loading Wikipedia (target: 150k sentences)...")
    try:
        wiki = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)
        wiki_count = 0
        wiki_target = 150000

        for item in wiki:
            if wiki_count >= wiki_target:
                break

            text = item.get('text', '')
            if text and len(text) > 50:  # Filter out very short texts
                # Split into sentences (simple approach)
                sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
                corpus.extend(sentences[:10])  # Take first 10 sentences per article
                wiki_count += len(sentences[:10])

        logger.info(f"✅ Wikipedia: {wiki_count} sentences collected")
    except Exception as e:
        logger.warning(f"⚠️  Wikipedia download failed: {e}")
        logger.info("Continuing with other sources...")

    # 2. Simple Wikipedia (backup, lighter dataset)
    if len(corpus) < 100000:
        logger.info("\n2️⃣ Loading Simple Wikipedia as backup...")
        try:
            from datasets import load_dataset
            simple_wiki = load_dataset("wikipedia", "20220301.simple", split="train[:50000]", trust_remote_code=True)

            for item in simple_wiki:
                text = item.get('text', '')
                if text and len(text) > 50:
                    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
                    corpus.extend(sentences[:5])

            logger.info(f"✅ Simple Wikipedia: {len(corpus)} total sentences now")
        except Exception as e:
            logger.warning(f"⚠️  Simple Wikipedia failed: {e}")

    # 3. SQuAD Q&A (20% - conversational/question patterns)
    logger.info("\n3️⃣ Loading SQuAD Q&A (target: 100k sentences)...")
    try:
        squad = load_dataset("squad", split="train")

        qa_count = 0
        for item in squad:
            if qa_count >= 100000:
                break

            # Add questions
            question = item.get('question', '')
            if question:
                corpus.append(question)
                qa_count += 1

            # Add contexts (passages)
            context = item.get('context', '')
            if context and len(context) > 50:
                sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 20]
                corpus.extend(sentences[:3])
                qa_count += len(sentences[:3])

        logger.info(f"✅ SQuAD: {qa_count} sentences collected")
    except Exception as e:
        logger.warning(f"⚠️  SQuAD download failed: {e}")

    # 4. IMDB Reviews (10% - sentiment/opinions)
    logger.info("\n4️⃣ Loading IMDB reviews (target: 50k sentences)...")
    try:
        imdb = load_dataset("imdb", split="train[:10000]", trust_remote_code=True)

        review_count = 0
        for item in imdb:
            if review_count >= 50000:
                break

            text = item.get('text', '')
            if text and len(text) > 50:
                sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
                corpus.extend(sentences[:5])
                review_count += len(sentences[:5])

        logger.info(f"✅ IMDB: {review_count} sentences collected")
    except Exception as e:
        logger.warning(f"⚠️  IMDB download failed: {e}")

    # 5. BookCorpus samples (10% - literary/narrative text)
    logger.info("\n5️⃣ Loading BookCorpus samples (target: 50k sentences)...")
    try:
        books = load_dataset("bookcorpus", split="train", streaming=True, trust_remote_code=True)

        book_count = 0
        for item in books:
            if book_count >= 50000:
                break

            text = item.get('text', '')
            if text and len(text) > 30:
                corpus.append(text.strip())
                book_count += 1

        logger.info(f"✅ BookCorpus: {book_count} sentences collected")
    except Exception as e:
        logger.warning(f"⚠️  BookCorpus download failed: {e}")

    # Clean and deduplicate
    logger.info("\n🧹 Cleaning corpus...")

    # Remove duplicates
    corpus_unique = list(set(corpus))
    logger.info(f"  Removed {len(corpus) - len(corpus_unique)} duplicates")

    # Filter by length (20-500 characters)
    corpus_filtered = [s for s in corpus_unique if 20 <= len(s) <= 500]
    logger.info(f"  Filtered by length: {len(corpus_filtered)} sentences remaining")

    # Shuffle
    random.shuffle(corpus_filtered)

    # Limit to 500k for efficiency
    corpus_final = corpus_filtered[:500000]

    logger.info(f"\n📊 Final corpus statistics:")
    logger.info(f"  Total sentences: {len(corpus_final)}")
    logger.info(f"  Avg length: {sum(len(s) for s in corpus_final) / len(corpus_final):.1f} chars")
    logger.info(f"  Min length: {min(len(s) for s in corpus_final)} chars")
    logger.info(f"  Max length: {max(len(s) for s in corpus_final)} chars")

    # Save corpus
    logger.info(f"\n💾 Saving corpus to {output_dir}...")

    # Save as JSONL (one sentence per line)
    corpus_file = output_dir / "corpus.jsonl"
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for sentence in corpus_final:
            json.dump({"text": sentence}, f, ensure_ascii=False)
            f.write('\n')

    # Save as plain text (for quick inspection)
    corpus_txt = output_dir / "corpus_sample.txt"
    with open(corpus_txt, 'w', encoding='utf-8') as f:
        for sentence in corpus_final[:1000]:  # First 1000 for inspection
            f.write(sentence + '\n')

    corpus_size_mb = corpus_file.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Corpus saved: {corpus_size_mb:.1f} MB")

    logger.info("\n" + "=" * 80)
    logger.info("✅ CORPUS PREPARATION COMPLETE")
    logger.info("=" * 80)

    logger.info(f"\n📁 Files created:")
    logger.info(f"  - {corpus_file} (full corpus, {corpus_size_mb:.1f} MB)")
    logger.info(f"  - {corpus_txt} (sample for inspection)")

    logger.info(f"\n🚀 Next step:")
    logger.info(f"  python3 distill_model2vec_optimized.py")

    return True


def main():
    """Main execution"""
    import sys

    success = prepare_corpus()

    if success:
        logger.info("\n🎉 Corpus preparation successful!")
        sys.exit(0)
    else:
        logger.error("\n❌ Corpus preparation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
