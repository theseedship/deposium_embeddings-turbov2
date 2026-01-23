"""
Pydantic models for Anthropic Messages API compatibility.

Based on: https://docs.anthropic.com/en/api/messages
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
import uuid
import time


# =============================================================================
# Content Blocks (used in messages)
# =============================================================================

class TextBlock(BaseModel):
    """Text content block."""
    type: Literal["text"] = "text"
    text: str


class ImageSource(BaseModel):
    """Image source for image content blocks."""
    type: Literal["base64"] = "base64"
    media_type: str = Field(..., description="MIME type (e.g., image/jpeg)")
    data: str = Field(..., description="Base64-encoded image data")


class ImageBlock(BaseModel):
    """Image content block."""
    type: Literal["image"] = "image"
    source: ImageSource


class ToolUseBlock(BaseModel):
    """Tool use content block (assistant requesting tool execution)."""
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(default_factory=lambda: f"toolu_{uuid.uuid4().hex[:24]}")
    name: str
    input: Dict[str, Any]


class ToolResultBlock(BaseModel):
    """Tool result content block (user providing tool output)."""
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List["TextBlock"]]
    is_error: bool = False


# Union type for all content blocks
ContentBlock = Union[TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock]


# =============================================================================
# Tool Definitions
# =============================================================================

class ToolInputSchema(BaseModel):
    """JSON Schema for tool input."""
    type: Literal["object"] = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)


class Tool(BaseModel):
    """Tool definition for function calling."""
    name: str
    description: str
    input_schema: ToolInputSchema


# =============================================================================
# Messages
# =============================================================================

class AnthropicMessage(BaseModel):
    """A message in the conversation."""
    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]

    def get_text_content(self) -> str:
        """Extract text content from message."""
        if isinstance(self.content, str):
            return self.content

        text_parts = []
        for block in self.content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, str):
                        text_parts.append(content)
            elif hasattr(block, "type"):
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_result":
                    if isinstance(block.content, str):
                        text_parts.append(block.content)

        return "\n".join(text_parts)


# =============================================================================
# API Request/Response Models
# =============================================================================

class AnthropicMessagesRequest(BaseModel):
    """Request body for /v1/messages endpoint."""
    model: str = Field(..., description="Model identifier")
    messages: List[AnthropicMessage] = Field(..., description="Conversation messages")
    max_tokens: int = Field(default=1024, ge=1, le=32768, description="Maximum tokens to generate")
    system: Optional[str] = Field(default=None, description="System prompt")
    stream: bool = Field(default=False, description="Enable SSE streaming")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-k sampling")
    tools: Optional[List[Tool]] = Field(default=None, description="Available tools")
    tool_choice: Optional[Dict[str, Any]] = Field(default=None, description="Tool choice strategy")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Stop sequences")

    # Anthropic-specific fields we accept but may not fully support
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Request metadata")


class Usage(BaseModel):
    """Token usage statistics."""
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicMessagesResponse(BaseModel):
    """Response body for /v1/messages endpoint."""
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[ContentBlock]
    model: str
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", None] = None
    stop_sequence: Optional[str] = None
    usage: Usage = Field(default_factory=Usage)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello! How can I help you?"}],
                "model": "qwen2.5-coder-7b",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 8}
            }
        }


# =============================================================================
# SSE Streaming Events
# =============================================================================

class MessageStartEvent(BaseModel):
    """Initial event with message metadata."""
    type: Literal["message_start"] = "message_start"
    message: Dict[str, Any]


class ContentBlockStartEvent(BaseModel):
    """Start of a content block."""
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: Dict[str, Any]


class ContentBlockDeltaEvent(BaseModel):
    """Delta update to a content block."""
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: Dict[str, Any]


class ContentBlockStopEvent(BaseModel):
    """End of a content block."""
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaEvent(BaseModel):
    """Update to message-level fields."""
    type: Literal["message_delta"] = "message_delta"
    delta: Dict[str, Any]
    usage: Usage


class MessageStopEvent(BaseModel):
    """Final event indicating message completion."""
    type: Literal["message_stop"] = "message_stop"


class PingEvent(BaseModel):
    """Keep-alive ping event."""
    type: Literal["ping"] = "ping"


class ErrorEvent(BaseModel):
    """Error event."""
    type: Literal["error"] = "error"
    error: Dict[str, Any]


# Union of all SSE events
SSEEvent = Union[
    MessageStartEvent,
    ContentBlockStartEvent,
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStopEvent,
    PingEvent,
    ErrorEvent,
]
