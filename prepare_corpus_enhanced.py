#!/usr/bin/env python3
"""
Enhanced Corpus Preparation with Scientific/Technical Domains

Adds specialized datasets:
- Scientific papers (arXiv)
- Medical/biomedical (PubMed)
- Mathematics
- Computer Science
- Physics, Chemistry, Biology

Total target: ~500k sentences across diverse scientific & general domains
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


def prepare_enhanced_corpus():
    """Prepare enhanced corpus with scientific domains"""

    logger.info("=" * 80)
    logger.info("🔬 Preparing ENHANCED Corpus with Scientific Domains")
    logger.info("=" * 80)

    try:
        from datasets import load_dataset
        logger.info("✅ datasets library imported")
    except ImportError:
        logger.error("❌ datasets library not installed")
        return False

    output_dir = Path("./data/model2vec_corpus_enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing general corpus
    existing_corpus_path = Path("./data/model2vec_corpus/corpus.jsonl")
    corpus: List[str] = []

    if existing_corpus_path.exists():
        logger.info(f"\n📂 Loading existing general corpus...")
        with open(existing_corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                corpus.append(data['text'])
        logger.info(f"✅ Loaded {len(corpus)} sentences from general corpus")
    else:
        logger.warning("⚠️  No existing corpus found, starting from scratch")

    logger.info("\n📥 Adding Scientific/Technical Domains...")

    # 6. Scientific Papers (arXiv abstracts)
    logger.info("\n6️⃣ Loading Scientific Papers (arXiv) (target: 100k sentences)...")
    try:
        arxiv = load_dataset("scientific_papers", "arxiv", split="train", streaming=True, trust_remote_code=True)

        arxiv_count = 0
        for item in arxiv:
            if arxiv_count >= 100000:
                break

            # Get abstract
            abstract = item.get('abstract', '')
            if abstract and len(abstract) > 50:
                # Split into sentences
                sentences = [s.strip() for s in abstract.split('.') if len(s.strip()) > 20]
                corpus.extend(sentences[:5])
                arxiv_count += len(sentences[:5])

        logger.info(f"✅ Scientific Papers: {arxiv_count} sentences collected")
    except Exception as e:
        logger.warning(f"⚠️  Scientific papers failed: {e}")

    # 7. Medical/Biomedical (PubMed abstracts)
    logger.info("\n7️⃣ Loading Medical Papers (PubMed) (target: 50k sentences)...")
    try:
        pubmed = load_dataset("scientific_papers", "pubmed", split="train", streaming=True, trust_remote_code=True)

        pubmed_count = 0
        for item in pubmed:
            if pubmed_count >= 50000:
                break

            abstract = item.get('abstract', '')
            if abstract and len(abstract) > 50:
                sentences = [s.strip() for s in abstract.split('.') if len(s.strip()) > 20]
                corpus.extend(sentences[:3])
                pubmed_count += len(sentences[:3])

        logger.info(f"✅ Medical Papers: {pubmed_count} sentences collected")
    except Exception as e:
        logger.warning(f"⚠️  PubMed failed: {e}")

    # 8. Science Q&A (general science)
    logger.info("\n8️⃣ Loading Science Q&A (target: 25k sentences)...")
    try:
        sciq = load_dataset("sciq", split="train", trust_remote_code=True)

        sci_count = 0
        for item in sciq:
            if sci_count >= 25000:
                break

            # Questions and support text
            question = item.get('question', '')
            support = item.get('support', '')

            if question:
                corpus.append(question)
                sci_count += 1
            if support and len(support) > 30:
                corpus.append(support)
                sci_count += 1

        logger.info(f"✅ Science Q&A: {sci_count} sentences collected")
    except Exception as e:
        logger.warning(f"⚠️  Science Q&A failed: {e}")

    # 9. Mathematics Q&A
    logger.info("\n9️⃣ Loading Math Q&A (target: 25k sentences)...")
    try:
        math_qa = load_dataset("math_qa", split="train", trust_remote_code=True)

        math_count = 0
        for item in math_qa:
            if math_count >= 25000:
                break

            problem = item.get('Problem', '') or item.get('problem', '')
            if problem and len(problem) > 20:
                corpus.append(problem)
                math_count += 1

        logger.info(f"✅ Math Q&A: {math_count} sentences collected")
    except Exception as e:
        logger.warning(f"⚠️  Math Q&A failed: {e}")

    # 10. Computer Science (code documentation)
    logger.info("\n🔟 Loading CS Documentation (target: 25k sentences)...")
    try:
        code_docs = load_dataset("code_search_net", "python", split="train", streaming=True, trust_remote_code=True)

        code_count = 0
        for item in code_docs:
            if code_count >= 25000:
                break

            docstring = item.get('func_documentation_string', '') or item.get('docstring', '')
            if docstring and len(docstring) > 30:
                corpus.append(docstring.strip())
                code_count += 1

        logger.info(f"✅ CS Documentation: {code_count} sentences collected")
    except Exception as e:
        logger.warning(f"⚠️  CS docs failed: {e}")

    # Clean and deduplicate
    logger.info("\n🧹 Cleaning enhanced corpus...")

    # Remove duplicates
    corpus_unique = list(set(corpus))
    logger.info(f"  Removed {len(corpus) - len(corpus_unique)} duplicates")

    # Filter by length
    corpus_filtered = [s for s in corpus_unique if 20 <= len(s) <= 500]
    logger.info(f"  Filtered by length: {len(corpus_filtered)} sentences remaining")

    # Shuffle
    random.shuffle(corpus_filtered)

    # Limit to 500k
    corpus_final = corpus_filtered[:500000]

    logger.info(f"\n📊 Final enhanced corpus statistics:")
    logger.info(f"  Total sentences: {len(corpus_final)}")
    logger.info(f"  Avg length: {sum(len(s) for s in corpus_final) / len(corpus_final):.1f} chars")
    logger.info(f"  Min length: {min(len(s) for s in corpus_final)} chars")
    logger.info(f"  Max length: {max(len(s) for s in corpus_final)} chars")

    # Domain breakdown estimate
    logger.info(f"\n📈 Estimated domain distribution:")
    logger.info(f"  General (Wiki, Books, etc): ~{min(288000, len(corpus_final))} sentences")
    logger.info(f"  Scientific (arXiv, PubMed): ~{min(150000, len(corpus_final) - 288000)} sentences")
    logger.info(f"  Technical (Math, CS, Sci): ~{min(75000, len(corpus_final) - 438000)} sentences")

    # Save corpus
    logger.info(f"\n💾 Saving enhanced corpus...")

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
    logger.info(f"✅ Enhanced corpus saved: {corpus_size_mb:.1f} MB")

    logger.info("\n" + "=" * 80)
    logger.info("✅ ENHANCED CORPUS PREPARATION COMPLETE")
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

    success = prepare_enhanced_corpus()

    if success:
        logger.info("\n🎉 Enhanced corpus preparation successful!")
        sys.exit(0)
    else:
        logger.error("\n❌ Enhanced corpus preparation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
