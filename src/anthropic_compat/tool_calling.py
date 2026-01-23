"""
Tool calling support for Anthropic-compatible API.

Handles parsing tool calls from model output and converting them
to Anthropic's tool_use content blocks.
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
from .schemas import TextBlock, ToolUseBlock, ContentBlock


# Regex patterns for tool call parsing
TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*({.*?})\s*</tool_call>',
    re.DOTALL
)

# Alternative patterns some models might use
ALT_TOOL_PATTERNS = [
    re.compile(r'```tool_call\s*\n({.*?})\n```', re.DOTALL),
    re.compile(r'\[TOOL_CALL\]\s*({.*?})\s*\[/TOOL_CALL\]', re.DOTALL),
    # Qwen function calling format
    re.compile(r'<\|tool_call\|>\s*({.*?})\s*(?:<\|/tool_call\|>|$)', re.DOTALL),
]


def parse_tool_calls(
    text: str
) -> Tuple[List[ContentBlock], bool, str]:
    """
    Parse tool calls from model output text.

    Args:
        text: Raw model output text

    Returns:
        Tuple of:
        - List of content blocks (text and tool_use blocks)
        - Whether any tool calls were found
        - Remaining text after tool calls
    """
    content_blocks: List[ContentBlock] = []
    has_tool_calls = False

    # Try main pattern first
    tool_matches = list(TOOL_CALL_PATTERN.finditer(text))

    # Try alternative patterns if main pattern fails
    if not tool_matches:
        for pattern in ALT_TOOL_PATTERNS:
            tool_matches = list(pattern.finditer(text))
            if tool_matches:
                break

    if not tool_matches:
        # No tool calls found - return text as-is
        if text.strip():
            content_blocks.append(TextBlock(text=text.strip()))
        return content_blocks, False, text

    # Process text and tool calls in order
    last_end = 0

    for match in tool_matches:
        # Add any text before this tool call
        text_before = text[last_end:match.start()].strip()
        if text_before:
            content_blocks.append(TextBlock(text=text_before))

        # Parse the tool call JSON
        try:
            tool_json = match.group(1)
            tool_data = json.loads(tool_json)

            # Extract tool name and arguments
            tool_name = tool_data.get("name") or tool_data.get("function", {}).get("name")
            tool_args = tool_data.get("arguments") or tool_data.get("input") or tool_data.get("parameters", {})

            # Handle string arguments (some models double-encode)
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {"raw_input": tool_args}

            if tool_name:
                content_blocks.append(ToolUseBlock(
                    id=f"toolu_{uuid.uuid4().hex[:24]}",
                    name=tool_name,
                    input=tool_args or {}
                ))
                has_tool_calls = True

        except json.JSONDecodeError as e:
            # If JSON parsing fails, include the raw text
            content_blocks.append(TextBlock(
                text=f"[Tool call parsing error: {e}]\n{match.group(0)}"
            ))

        last_end = match.end()

    # Add any remaining text after the last tool call
    remaining_text = text[last_end:].strip()
    if remaining_text:
        content_blocks.append(TextBlock(text=remaining_text))

    return content_blocks, has_tool_calls, remaining_text


def extract_tool_call_streaming(
    buffer: str
) -> Tuple[Optional[Dict[str, Any]], str, bool]:
    """
    Extract a complete tool call from a streaming buffer.

    Used during streaming to detect when a complete tool call
    has been generated.

    Args:
        buffer: Accumulated text buffer

    Returns:
        Tuple of:
        - Parsed tool call dict (if complete) or None
        - Updated buffer (text before tool call, or full buffer if incomplete)
        - Whether we're in the middle of a tool call
    """
    # Check if we're starting a tool call
    tool_start_markers = ["<tool_call>", "```tool_call", "[TOOL_CALL]", "<|tool_call|>"]

    in_tool_call = False
    for marker in tool_start_markers:
        if marker in buffer:
            in_tool_call = True
            break

    if not in_tool_call:
        return None, buffer, False

    # Try to extract complete tool call
    for pattern in [TOOL_CALL_PATTERN] + ALT_TOOL_PATTERNS:
        match = pattern.search(buffer)
        if match:
            try:
                tool_data = json.loads(match.group(1))
                text_before = buffer[:match.start()].strip()

                tool_name = tool_data.get("name") or tool_data.get("function", {}).get("name")
                tool_args = tool_data.get("arguments") or tool_data.get("input") or tool_data.get("parameters", {})

                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {"raw_input": tool_args}

                return {
                    "name": tool_name,
                    "arguments": tool_args or {}
                }, text_before, False

            except json.JSONDecodeError:
                # Incomplete JSON, keep buffering
                pass

    # Still accumulating tool call
    return None, buffer, True


def format_tool_result_for_model(
    tool_use_id: str,
    result: str,
    is_error: bool = False
) -> str:
    """
    Format a tool result for inclusion in the conversation.

    Args:
        tool_use_id: ID of the tool use this is responding to
        result: Tool execution result or error message
        is_error: Whether this is an error result

    Returns:
        Formatted string for the model
    """
    if is_error:
        return f'<tool_error id="{tool_use_id}">\n{result}\n</tool_error>'
    else:
        return f'<tool_result id="{tool_use_id}">\n{result}\n</tool_result>'
