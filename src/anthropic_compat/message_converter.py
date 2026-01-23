"""
Convert between Anthropic message format and HuggingFace chat format.

Handles the transformation of Anthropic-style messages to the format
expected by HuggingFace chat templates (Qwen, Llama, Mistral, etc.).
"""

from typing import Any, Dict, List, Optional
from .schemas import AnthropicMessage, Tool


def convert_anthropic_to_hf_messages(
    messages: List[AnthropicMessage],
    system: Optional[str] = None,
    tools: Optional[List[Tool]] = None,
) -> List[Dict[str, Any]]:
    """
    Convert Anthropic messages to HuggingFace chat format.

    Args:
        messages: List of Anthropic messages
        system: Optional system prompt
        tools: Optional tool definitions (will be injected into system prompt)

    Returns:
        List of messages in HuggingFace format
    """
    hf_messages = []

    # Build system prompt with tools if provided
    system_content = _build_system_prompt(system, tools)
    if system_content:
        hf_messages.append({
            "role": "system",
            "content": system_content
        })

    # Convert each message
    for msg in messages:
        hf_msg = _convert_message(msg)
        if hf_msg:
            hf_messages.append(hf_msg)

    return hf_messages


def _build_system_prompt(
    system: Optional[str] = None,
    tools: Optional[List[Tool]] = None
) -> str:
    """
    Build the system prompt, optionally including tool definitions.

    Args:
        system: Base system prompt
        tools: Tool definitions to include

    Returns:
        Combined system prompt
    """
    parts = []

    if system:
        parts.append(system)

    if tools:
        tools_text = _format_tools_for_prompt(tools)
        parts.append(tools_text)

    return "\n\n".join(parts)


def _format_tools_for_prompt(tools: List[Tool]) -> str:
    """
    Format tool definitions for injection into the system prompt.

    Uses a format that Qwen2.5-Coder and similar models understand.
    """
    tools_description = ["You have access to the following tools:"]

    for tool in tools:
        tool_text = f"""
<tool>
Name: {tool.name}
Description: {tool.description}
Parameters: {_format_json_schema(tool.input_schema.model_dump())}
</tool>"""
        tools_description.append(tool_text)

    tools_description.append("""
When you need to use a tool, respond with:
<tool_call>
{"name": "tool_name", "arguments": {"param1": "value1", ...}}
</tool_call>

Wait for the tool result before continuing. Do not make assumptions about tool outputs.""")

    return "\n".join(tools_description)


def _format_json_schema(schema: Dict[str, Any]) -> str:
    """Format a JSON schema for display in prompt."""
    import json
    return json.dumps(schema, indent=2)


def _convert_message(msg: AnthropicMessage) -> Optional[Dict[str, Any]]:
    """
    Convert a single Anthropic message to HuggingFace format.

    Args:
        msg: Anthropic message

    Returns:
        HuggingFace message dict or None
    """
    # Handle string content directly
    if isinstance(msg.content, str):
        return {
            "role": msg.role,
            "content": msg.content
        }

    # Handle list of content blocks
    content_parts = []

    for block in msg.content:
        # Handle dict-style blocks (from JSON parsing)
        if isinstance(block, dict):
            block_type = block.get("type")

            if block_type == "text":
                content_parts.append(block.get("text", ""))

            elif block_type == "tool_use":
                # Assistant requesting tool use - format as tool call
                import json
                tool_call = {
                    "name": block.get("name"),
                    "arguments": block.get("input", {})
                }
                content_parts.append(f"<tool_call>\n{json.dumps(tool_call, indent=2)}\n</tool_call>")

            elif block_type == "tool_result":
                # User providing tool result
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    result_content = "\n".join(
                        b.get("text", "") for b in result_content if b.get("type") == "text"
                    )
                tool_use_id = block.get("tool_use_id", "unknown")
                is_error = block.get("is_error", False)

                if is_error:
                    content_parts.append(f"<tool_error id=\"{tool_use_id}\">\n{result_content}\n</tool_error>")
                else:
                    content_parts.append(f"<tool_result id=\"{tool_use_id}\">\n{result_content}\n</tool_result>")

        # Handle Pydantic model blocks
        elif hasattr(block, "type"):
            if block.type == "text":
                content_parts.append(block.text)

            elif block.type == "tool_use":
                import json
                tool_call = {
                    "name": block.name,
                    "arguments": block.input
                }
                content_parts.append(f"<tool_call>\n{json.dumps(tool_call, indent=2)}\n</tool_call>")

            elif block.type == "tool_result":
                result_content = block.content
                if isinstance(result_content, list):
                    result_content = "\n".join(b.text for b in result_content if hasattr(b, "text"))

                if block.is_error:
                    content_parts.append(f"<tool_error id=\"{block.tool_use_id}\">\n{result_content}\n</tool_error>")
                else:
                    content_parts.append(f"<tool_result id=\"{block.tool_use_id}\">\n{result_content}\n</tool_result>")

    if not content_parts:
        return None

    return {
        "role": msg.role,
        "content": "\n".join(content_parts)
    }
