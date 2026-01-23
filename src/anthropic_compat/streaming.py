"""
SSE Streaming support for Anthropic-compatible API.

Generates Server-Sent Events in the format expected by
Anthropic clients (Claude Code, etc.).
"""

import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional
from .schemas import (
    AnthropicMessagesRequest,
    ContentBlock,
    TextBlock,
    ToolUseBlock,
    Usage,
)
from .tool_calling import extract_tool_call_streaming

logger = logging.getLogger(__name__)


def format_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """
    Format a Server-Sent Event.

    Args:
        event_type: The event type
        data: The event data

    Returns:
        Formatted SSE string
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def generate_sse_stream(
    request: AnthropicMessagesRequest,
    text_generator: Generator,
    model_name: str,
) -> AsyncGenerator[str, None]:
    """
    Generate SSE events from a text generator.

    Follows Anthropic's streaming format:
    1. message_start - Initial message metadata
    2. content_block_start - Start of each content block
    3. content_block_delta - Text chunks
    4. content_block_stop - End of content block
    5. message_delta - Final usage stats
    6. message_stop - Stream complete

    Args:
        request: The original request
        text_generator: Generator yielding (text, is_final, input_tokens, output_tokens)
        model_name: Name of the model being used

    Yields:
        SSE-formatted strings
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Track state
    input_tokens = 0
    output_tokens = 0
    current_block_index = 0
    block_started = False
    accumulated_text = ""
    tool_call_buffer = ""
    in_tool_call = False
    has_tool_calls = False

    # 1. message_start event
    yield format_sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model_name,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}
        }
    })

    try:
        for text_chunk, is_final, in_tok, out_tok in text_generator:
            input_tokens = in_tok
            output_tokens = out_tok

            if is_final:
                break

            if not text_chunk:
                continue

            # Check for tool calls in streaming
            if request.tools:
                tool_call_buffer += text_chunk
                tool_call, text_before, still_in_tool = extract_tool_call_streaming(tool_call_buffer)

                if still_in_tool:
                    in_tool_call = True
                    continue  # Keep buffering

                if tool_call:
                    # Output any text before the tool call
                    if text_before and not block_started:
                        yield format_sse_event("content_block_start", {
                            "type": "content_block_start",
                            "index": current_block_index,
                            "content_block": {"type": "text", "text": ""}
                        })
                        block_started = True

                    if text_before:
                        yield format_sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": current_block_index,
                            "delta": {"type": "text_delta", "text": text_before}
                        })
                        accumulated_text += text_before

                    # Close text block if open
                    if block_started:
                        yield format_sse_event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": current_block_index
                        })
                        current_block_index += 1
                        block_started = False

                    # Emit tool use block
                    tool_use_id = f"toolu_{uuid.uuid4().hex[:24]}"
                    yield format_sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": current_block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": tool_call["name"],
                            "input": {}
                        }
                    })

                    # Send input as delta
                    yield format_sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": current_block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(tool_call["arguments"])
                        }
                    })

                    yield format_sse_event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": current_block_index
                    })

                    current_block_index += 1
                    has_tool_calls = True
                    tool_call_buffer = ""
                    in_tool_call = False
                    continue

                # No tool call found, process normally
                tool_call_buffer = ""
                in_tool_call = False

            # Start content block if needed
            if not block_started:
                yield format_sse_event("content_block_start", {
                    "type": "content_block_start",
                    "index": current_block_index,
                    "content_block": {"type": "text", "text": ""}
                })
                block_started = True

            # Emit text delta
            yield format_sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": current_block_index,
                "delta": {"type": "text_delta", "text": text_chunk}
            })
            accumulated_text += text_chunk

        # Close any open block
        if block_started:
            yield format_sse_event("content_block_stop", {
                "type": "content_block_stop",
                "index": current_block_index
            })

        # Determine stop reason
        stop_reason = "tool_use" if has_tool_calls else "end_turn"

        # 5. message_delta event
        yield format_sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens}
        })

        # 6. message_stop event
        yield format_sse_event("message_stop", {
            "type": "message_stop"
        })

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield format_sse_event("error", {
            "type": "error",
            "error": {
                "type": "server_error",
                "message": str(e)
            }
        })


def create_ping_event() -> str:
    """Create a ping event for keep-alive."""
    return format_sse_event("ping", {"type": "ping"})
