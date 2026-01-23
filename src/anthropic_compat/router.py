"""
FastAPI router for Anthropic Messages API compatibility.

Provides the /v1/messages endpoint that mimics Anthropic's API,
allowing Claude Code and other Anthropic-compatible clients to
work with local LLMs.
"""

import logging
import time
import uuid
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from .schemas import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    ContentBlock,
    TextBlock,
    ToolUseBlock,
    Usage,
)
from .message_converter import convert_anthropic_to_hf_messages
from .tool_calling import parse_tool_calls
from .streaming import generate_sse_stream
from .llm_backend import LLMBackend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Anthropic Compatibility"])

# Will be set by main.py when mounting the router
_model_manager = None
_verify_api_key = None


def set_dependencies(model_manager, verify_api_key):
    """Set dependencies from main application."""
    global _model_manager, _verify_api_key
    _model_manager = model_manager
    _verify_api_key = verify_api_key


def get_model_manager():
    """Get the model manager instance."""
    if _model_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Model manager not initialized"
        )
    return _model_manager


async def verify_key(request: Request):
    """Verify API key if configured."""
    if _verify_api_key is not None:
        return await _verify_api_key(request)
    return "no-auth"


@router.post("/messages", response_model=AnthropicMessagesResponse)
async def create_message(
    request: AnthropicMessagesRequest,
    api_key: str = Depends(verify_key)
):
    """
    Create a message using the Anthropic-compatible API.

    This endpoint mimics Anthropic's /v1/messages API, allowing
    Claude Code and other clients to use local LLMs.

    **Supported features:**
    - Text generation with chat history
    - System prompts
    - Temperature, top_p, top_k sampling
    - Tool/function calling
    - Streaming (SSE)

    **Example:**
    ```bash
    curl -X POST http://localhost:8000/v1/messages \\
      -H "Authorization: Bearer $API_KEY" \\
      -H "Content-Type: application/json" \\
      -d '{
        "model": "qwen2.5-coder-7b",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello!"}]
      }'
    ```
    """
    model_manager = get_model_manager()

    # Validate model exists and is a causal LM
    if request.model not in model_manager.configs:
        available_llms = [
            name for name, cfg in model_manager.configs.items()
            if cfg.type == "causal_lm"
        ]
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found. Available LLMs: {available_llms}"
        )

    config = model_manager.configs[request.model]
    if config.type != "causal_lm":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is not a causal language model (type: {config.type})"
        )

    try:
        # Get the model (lazy loading)
        model, tokenizer = model_manager.get_model(request.model)

        # Create LLM backend
        backend = LLMBackend(model, tokenizer, device=config.device)

        # Convert messages to HuggingFace format
        hf_messages = convert_anthropic_to_hf_messages(
            messages=request.messages,
            system=request.system,
            tools=request.tools
        )

        logger.info(f"Processing request for model {request.model}, {len(request.messages)} messages")

        # Handle streaming
        if request.stream:
            # Create streaming response
            def sync_generator():
                yield from backend.generate_stream(
                    messages=hf_messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    stop_sequences=request.stop_sequences,
                )

            async def async_wrapper():
                async for event in generate_sse_stream(
                    request=request,
                    text_generator=sync_generator(),
                    model_name=request.model,
                ):
                    yield event

            return StreamingResponse(
                async_wrapper(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )

        # Non-streaming generation
        start_time = time.time()

        generated_text, input_tokens, output_tokens = backend.generate(
            messages=hf_messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_sequences=request.stop_sequences,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Generated {output_tokens} tokens in {elapsed_ms:.0f}ms")

        # Parse tool calls if tools were provided
        content_blocks: list[ContentBlock] = []
        stop_reason = "end_turn"

        if request.tools:
            content_blocks, has_tool_calls, _ = parse_tool_calls(generated_text)
            if has_tool_calls:
                stop_reason = "tool_use"
        else:
            content_blocks = [TextBlock(text=generated_text)]

        # Check if we hit max tokens
        if output_tokens >= request.max_tokens:
            stop_reason = "max_tokens"

        return AnthropicMessagesResponse(
            id=f"msg_{uuid.uuid4().hex[:24]}",
            content=content_blocks,
            model=request.model,
            stop_reason=stop_reason,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        )

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


@router.get("/models")
async def list_models(api_key: str = Depends(verify_key)):
    """
    List available models.

    Returns models that support the Anthropic Messages API
    (causal language models).
    """
    model_manager = get_model_manager()

    models = []
    for name, config in model_manager.configs.items():
        if config.type == "causal_lm":
            models.append({
                "id": name,
                "object": "model",
                "created": 1700000000,
                "owned_by": "local",
                "context_length": getattr(config, "context_length", 4096),
                "capabilities": {
                    "tool_use": True,
                    "streaming": True,
                }
            })

    return {"data": models, "object": "list"}
