#!/usr/bin/env python3
"""
Inspect ColBERT Model Specifications
- Embedding dimensions
- Max sequence length
- Model architecture details
"""

import logging
from pylate import models

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("🔍 Inspection: mxbai-edge-colbert-v0-32m")
    logger.info("=" * 80)

    # Load model
    logger.info("\n📥 Loading model...")
    model = models.ColBERT(
        model_name_or_path="mixedbread-ai/mxbai-edge-colbert-v0-32m",
    )
    logger.info("✅ Model loaded\n")

    # Get model config
    logger.info("📊 Model Specifications:")
    logger.info("-" * 80)

    # Check if model has a config
    if hasattr(model, 'model'):
        if hasattr(model.model, 'config'):
            config = model.model.config
            logger.info(f"Model Type: {config.model_type if hasattr(config, 'model_type') else 'N/A'}")
            logger.info(f"Hidden Size (embedding dim): {config.hidden_size if hasattr(config, 'hidden_size') else 'N/A'}")
            logger.info(f"Max Position Embeddings (context length): {config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else 'N/A'}")
            logger.info(f"Vocab Size: {config.vocab_size if hasattr(config, 'vocab_size') else 'N/A'}")
            logger.info(f"Num Hidden Layers: {config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 'N/A'}")
            logger.info(f"Num Attention Heads: {config.num_attention_heads if hasattr(config, 'num_attention_heads') else 'N/A'}")

    # Test with actual encoding to get embedding shape
    logger.info("\n" + "-" * 80)
    logger.info("🧪 Testing Actual Encoding:")
    logger.info("-" * 80)

    test_text = "This is a test sentence to check embedding dimensions."
    embeddings = model.encode([test_text], is_query=True)

    logger.info(f"Input: '{test_text}'")
    logger.info(f"Output shape: {embeddings[0].shape}")
    logger.info(f"  → Num tokens: {embeddings[0].shape[0]}")
    logger.info(f"  → Embedding dimension per token: {embeddings[0].shape[1]}")

    # Test with longer text
    long_text = " ".join(["word"] * 100)
    long_embeddings = model.encode([long_text], is_query=True)

    logger.info(f"\nLong text (100 words): {long_embeddings[0].shape[0]} tokens")

    # Very long text to test max length
    very_long_text = " ".join(["word"] * 1000)
    very_long_embeddings = model.encode([very_long_text], is_query=True)

    logger.info(f"Very long text (1000 words): {very_long_embeddings[0].shape[0]} tokens")
    logger.info(f"  → Max sequence length: {very_long_embeddings[0].shape[0]} (likely truncated)")

    logger.info("\n" + "=" * 80)
    logger.info("📋 SUMMARY")
    logger.info("=" * 80)

    if hasattr(model, 'model') and hasattr(model.model, 'config'):
        config = model.model.config
        hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else embeddings[0].shape[1]
        max_length = config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else very_long_embeddings[0].shape[0]

        logger.info(f"Embedding Dimension per Token: {hidden_size}D")
        logger.info(f"Max Context Length: {max_length} tokens")
        logger.info(f"Architecture: Multi-vector (ColBERT)")
        logger.info(f"Output: N vectors × {hidden_size}D (where N = number of tokens)")

        logger.info("\n💡 For Model2Vec Distillation:")
        logger.info(f"  Target dimensions: {hidden_size}D (same as base)")
        logger.info(f"  Vocabulary size: {config.vocab_size if hasattr(config, 'vocab_size') else 'N/A'}")
        logger.info(f"  Strategy: Average token embeddings → single {hidden_size}D vector")

    logger.info("\n✅ Inspection complete!")

if __name__ == "__main__":
    main()
