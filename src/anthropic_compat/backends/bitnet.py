"""
BitNet inference backend for CPU-only 1-bit quantization.

Microsoft BitNet provides efficient CPU inference with:
- 1-bit quantization (1.58-bit weights)
- 2-6x speedup vs FP16
- 70-80% energy reduction
- Minimal memory footprint (~500MB for 2.4B model)

Requires BitNet installation with run_inference.py script.
https://github.com/microsoft/BitNet
"""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .base import (
    BackendCapabilities,
    GenerationConfig,
    GenerationResult,
    InferenceBackend,
    StreamChunk,
)
from .config import BitNetConfig

logger = logging.getLogger(__name__)


class BitNetBackend(InferenceBackend):
    """
    BitNet CPU-only inference backend.

    Uses Microsoft BitNet's run_inference.py script via subprocess
    for 1-bit quantized model inference. Ideal for CPU-only deployments
    like Railway or edge devices.

    Limitations:
    - No tool calling support
    - Subprocess overhead per generation
    - Approximate token counting (4 chars/token)
    - Pseudo-streaming via stdout parsing
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[BitNetConfig] = None,
    ):
        """
        Initialize the BitNet backend.

        Args:
            model_path: Path to GGUF model file (overrides config)
            config: BitNet configuration
        """
        self.config = config or BitNetConfig.from_env()

        # Model path from argument takes precedence
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = self.config.model_path

        self.bitnet_path = Path(self.config.bitnet_path)
        self.threads = self.config.threads
        self.context_length = self.config.context_length
        self.timeout = self.config.timeout

        # Verify paths
        self._inference_script = self.bitnet_path / "run_inference.py"

        self._capabilities = BackendCapabilities(
            streaming=True,  # Pseudo-streaming via stdout
            tool_calling=False,  # BitNet doesn't support tool calling
            batching=False,  # One request at a time
            multi_gpu=False,  # CPU only
            quantization=["1bit"],  # 1.58-bit quantization
            max_context_length=self.context_length,
            continuous_batching=False,
            paged_attention=False,
        )

        logger.info(
            f"BitNet backend initialized: model={self.model_path}, "
            f"threads={self.threads}, context={self.context_length}"
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return the capabilities of this backend."""
        return self._capabilities

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count.

        Note: BitNet doesn't expose tokenizer, using rough estimate.
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4

    def _build_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Convert chat messages to a single prompt string.

        Uses a simple format compatible with most instruction-tuned models.
        """
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")

        # Add assistant prefix for generation
        prompt_parts.append("Assistant:")

        return "".join(prompt_parts)

    def _run_inference(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> subprocess.CompletedProcess:
        """
        Run BitNet inference via subprocess.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Completed subprocess result
        """
        cmd = [
            "python",
            str(self._inference_script),
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "-t", str(self.threads),
            "-c", str(self.context_length),
            "--temp", str(temperature),
        ]

        logger.debug(f"Running BitNet command: {' '.join(cmd[:6])}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.bitnet_path),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            return result
        except subprocess.TimeoutExpired as e:
            logger.error(f"BitNet inference timeout after {self.timeout}s")
            raise TimeoutError(f"BitNet inference timeout after {self.timeout}s") from e
        except FileNotFoundError as e:
            logger.error(f"BitNet not found at {self.bitnet_path}")
            raise FileNotFoundError(
                f"BitNet installation not found at {self.bitnet_path}. "
                f"Expected script: {self._inference_script}"
            ) from e

    def _parse_output(self, stdout: str, stderr: str) -> str:
        """
        Parse BitNet output to extract generated text.

        BitNet outputs various info to stderr, generated text to stdout.
        """
        # Filter out common BitNet status messages
        lines = stdout.strip().split("\n")
        output_lines = []

        for line in lines:
            # Skip status/progress lines
            if any(skip in line.lower() for skip in [
                "loading model",
                "model loaded",
                "tokens/s",
                "inference time",
                "system info",
                "main:",
                "llama_",
            ]):
                continue
            output_lines.append(line)

        return "\n".join(output_lines).strip()

    def generate(
        self,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
    ) -> GenerationResult:
        """
        Generate a response synchronously.

        Args:
            messages: Chat messages
            config: Generation configuration

        Returns:
            GenerationResult with generated text and metadata
        """
        start_time = time.time()

        prompt = self._build_prompt(messages)
        input_tokens = self.count_tokens(prompt)

        result = self._run_inference(
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

        if result.returncode != 0:
            logger.error(f"BitNet error: {result.stderr}")
            raise RuntimeError(f"BitNet inference failed: {result.stderr}")

        # Parse output
        generated_text = self._parse_output(result.stdout, result.stderr)
        output_tokens = self.count_tokens(generated_text)

        # Check for stop sequences
        finish_reason = "stop"
        if config.stop_sequences:
            for stop_seq in config.stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
                    break

        if output_tokens >= config.max_tokens:
            finish_reason = "length"

        elapsed_ms = (time.time() - start_time) * 1000

        return GenerationResult(
            text=generated_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            latency_ms=elapsed_ms,
            tokens_per_second=output_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
        )

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
    ) -> Generator[StreamChunk, None, None]:
        """
        Generate a response with streaming.

        Note: BitNet doesn't support true streaming, so we run the full
        inference and emit chunks to simulate streaming behavior.

        Args:
            messages: Chat messages
            config: Generation configuration

        Yields:
            StreamChunk objects with text chunks
        """
        prompt = self._build_prompt(messages)
        input_tokens = self.count_tokens(prompt)

        # Build command for streaming (same as sync)
        cmd = [
            "python",
            str(self._inference_script),
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(config.max_tokens),
            "-t", str(self.threads),
            "-c", str(self.context_length),
            "--temp", str(config.temperature),
        ]

        try:
            # Use Popen for pseudo-streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.bitnet_path),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            output_tokens = 0
            accumulated_text = ""

            # Read stdout line by line for pseudo-streaming
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break

                # Skip status lines
                if any(skip in line.lower() for skip in [
                    "loading model",
                    "model loaded",
                    "tokens/s",
                    "inference time",
                    "system info",
                    "main:",
                    "llama_",
                ]):
                    continue

                # Emit chunk
                chunk_text = line
                output_tokens += self.count_tokens(chunk_text)
                accumulated_text += chunk_text

                # Check stop sequences
                should_stop = False
                if config.stop_sequences:
                    for stop_seq in config.stop_sequences:
                        if stop_seq in accumulated_text:
                            # Truncate at stop sequence
                            idx = accumulated_text.find(stop_seq)
                            chunk_text = accumulated_text[len(accumulated_text) - len(chunk_text):idx]
                            should_stop = True
                            break

                if chunk_text:
                    yield StreamChunk(
                        text=chunk_text,
                        is_final=False,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )

                if should_stop:
                    break

            # Wait for process to complete
            process.wait(timeout=self.timeout)

            # Check for errors
            if process.returncode != 0:
                stderr = process.stderr.read()
                logger.error(f"BitNet stream error: {stderr}")

            # Determine finish reason
            finish_reason = "stop"
            if output_tokens >= config.max_tokens:
                finish_reason = "length"

            # Final chunk
            yield StreamChunk(
                text="",
                is_final=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
            )

        except subprocess.TimeoutExpired:
            process.kill()
            logger.error(f"BitNet stream timeout after {self.timeout}s")
            yield StreamChunk(
                text="",
                is_final=True,
                input_tokens=input_tokens,
                output_tokens=0,
                finish_reason="timeout",
            )
        except FileNotFoundError:
            logger.error(f"BitNet not found at {self.bitnet_path}")
            raise FileNotFoundError(
                f"BitNet installation not found at {self.bitnet_path}"
            )

    def is_available(self) -> bool:
        """Check if BitNet is available and model exists."""
        # Check BitNet installation
        if not self._inference_script.exists():
            logger.warning(f"BitNet script not found: {self._inference_script}")
            return False

        # Check model file
        model_path = Path(self.model_path)
        if not model_path.exists():
            logger.warning(f"BitNet model not found: {self.model_path}")
            return False

        return True

    def shutdown(self) -> None:
        """Clean up resources (nothing to clean for subprocess-based backend)."""
        logger.info("BitNet backend shutdown complete")
