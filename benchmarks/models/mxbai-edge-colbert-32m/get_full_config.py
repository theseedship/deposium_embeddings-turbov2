#!/usr/bin/env python3
"""Get full model config and architecture details"""

from pylate import models
import json

# Load model
model = models.ColBERT(
    model_name_or_path="mixedbread-ai/mxbai-edge-colbert-v0-32m",
)

print("=" * 80)
print("📋 Full Model Configuration")
print("=" * 80)

# Try to access the underlying transformer model
if hasattr(model, 'model'):
    print("\n✅ Found model.model")

    if hasattr(model.model, 'config'):
        config = model.model.config
        print("\n📊 Config attributes:")
        print(f"  Model type: {getattr(config, 'model_type', 'N/A')}")
        print(f"  Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
        print(f"  Intermediate size: {getattr(config, 'intermediate_size', 'N/A')}")
        print(f"  Max position embeddings: {getattr(config, 'max_position_embeddings', 'N/A')}")
        print(f"  Vocab size: {getattr(config, 'vocab_size', 'N/A')}")
        print(f"  Num hidden layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
        print(f"  Num attention heads: {getattr(config, 'num_attention_heads', 'N/A')}")
        print(f"  Attention head size: {getattr(config, 'hidden_size', 0) // getattr(config, 'num_attention_heads', 1) if hasattr(config, 'num_attention_heads') else 'N/A'}")

        # Try to get the full config as dict
        if hasattr(config, 'to_dict'):
            print("\n" + "=" * 80)
            print("📄 Full Config JSON:")
            print("=" * 80)
            config_dict = config.to_dict()
            print(json.dumps(config_dict, indent=2))

    # Check for tokenizer
    if hasattr(model.model, 'tokenizer'):
        print("\n📝 Tokenizer info:")
        tokenizer = model.model.tokenizer
        print(f"  Model max length: {getattr(tokenizer, 'model_max_length', 'N/A')}")
        print(f"  Vocab size: {len(tokenizer) if hasattr(tokenizer, '__len__') else 'N/A'}")

# Check model layers
if hasattr(model, 'model'):
    print("\n" + "=" * 80)
    print("🏗️ Model Architecture:")
    print("=" * 80)

    for name, module in model.model.named_children():
        print(f"  {name}: {type(module).__name__}")

print("\n✅ Complete!")
