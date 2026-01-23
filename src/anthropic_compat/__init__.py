"""
Anthropic Messages API Compatibility Layer
==========================================

Provides a `/v1/messages` endpoint compatible with the Anthropic API,
enabling Claude Code and other Anthropic-compatible clients to use
local LLMs (like Qwen2.5-Coder) via this server.

Example usage:
    export ANTHROPIC_BASE_URL=http://localhost:8000
    export ANTHROPIC_API_KEY=your-api-key
    claude --model qwen2.5-coder-7b
"""

from .router import router as anthropic_router
from .schemas import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    ContentBlock,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    Usage,
)

__all__ = [
    "anthropic_router",
    "AnthropicMessage",
    "AnthropicMessagesRequest",
    "AnthropicMessagesResponse",
    "ContentBlock",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "Usage",
]
