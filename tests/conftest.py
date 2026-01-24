"""
Global pytest fixtures for all tests.

Provides mock models, tokenizers, and common test utilities.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def mock_model():
    """
    Create a mock HuggingFace model.

    Provides realistic mock behavior for:
    - config.max_position_embeddings
    - generate() method
    - quantization_config (optional)
    """
    model = MagicMock()

    # Model config
    model.config = MagicMock()
    model.config.max_position_embeddings = 4096
    model.config._name_or_path = "mock-model/test"

    # Generate returns tensor-like output
    mock_output = MagicMock()
    mock_output.__getitem__ = lambda self, idx: MagicMock(
        __getitem__=lambda s, i: [101, 102, 103, 104, 105]
    )
    model.generate.return_value = mock_output

    return model


@pytest.fixture
def mock_tokenizer():
    """
    Create a mock HuggingFace tokenizer.

    Provides realistic mock behavior for:
    - apply_chat_template()
    - encode() / decode()
    - pad_token_id / eos_token_id
    """
    tokenizer = MagicMock()

    # Chat template
    tokenizer.apply_chat_template.return_value = "<|user|>\nHello\n<|assistant|>\n"

    # Encoding/decoding
    tokenizer.encode.return_value = [101, 102, 103, 104, 105]
    tokenizer.decode.return_value = "Hello, world!"

    # Token IDs
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2

    # Callable tokenizer for inputs
    def tokenize_call(*args, **kwargs):
        result = MagicMock()
        result.__getitem__ = lambda self, key: MagicMock()
        result.to.return_value = result
        # Make input_ids indexable
        input_ids = MagicMock()
        input_ids.shape = [1, 10]  # batch_size=1, seq_len=10
        result.__getitem__ = lambda self, key: input_ids if key == "input_ids" else MagicMock()
        return result

    tokenizer.side_effect = tokenize_call
    tokenizer.return_value = tokenize_call()

    return tokenizer


@pytest.fixture
def sample_messages():
    """Sample chat messages in HuggingFace format."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]


@pytest.fixture
def sample_anthropic_messages():
    """Sample chat messages in Anthropic format."""
    return [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]
