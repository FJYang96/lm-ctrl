"""LLM client for constraint generation using Anthropic's Claude."""

import os
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv


class LLMClient:
    """Client for interacting with Claude API to generate optimization constraints."""

    def __init__(self) -> None:
        """Initialize the LLM client with environment configuration."""
        load_dotenv()

        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5-20251101")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "40000"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment variables. "
                "Please set your API key in the .env file."
            )

        self.client = Anthropic(api_key=self.api_key)

    def generate_constraints(
        self, system_prompt: str, user_message: str, context: str | None = None
    ) -> str:
        """
        Generate optimization constraints using Claude.

        Args:
            system_prompt: System prompt defining the task and output format
            user_message: User command or feedback for this iteration
            context: Previous iteration context and results

        Returns:
            Generated constraint code as a string
        """
        # Combine context and user message if context is provided
        full_user_message = user_message
        if context:
            full_user_message = f"{context}\n\n{user_message}"

        try:
            # Use streaming to handle long requests (required by new Anthropic SDK)
            response_text = ""
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": full_user_message}],
            ) as stream:
                for text in stream.text_stream:
                    response_text += text

            return response_text

        except Exception as e:
            print(f"Error calling Claude API: {e}")
            raise

    def generate_constraints_with_vision(
        self,
        system_prompt: str,
        user_message: str,
        context: str | None = None,
        images: list[str] | None = None,
    ) -> str:
        """
        Generate optimization constraints using Claude with optional image feedback.

        Args:
            system_prompt: System prompt defining the task and output format
            user_message: User command or feedback for this iteration
            context: Previous iteration context and results
            images: List of base64-encoded PNG images (trajectory frames)

        Returns:
            Generated constraint code as a string
        """
        full_user_message = user_message
        if context:
            full_user_message = f"{context}\n\n{user_message}"

        # Build message content with text and images
        content: list[dict[str, Any]] = []

        # Add images with labels (if provided)
        if images:
            # First half are planned trajectory frames, second half are simulated
            num_images = len(images)
            half = num_images // 2

            if half > 0:
                # Add label for planned trajectory frames
                content.append(
                    {
                        "type": "text",
                        "text": "PLANNED TRAJECTORY (what the optimizer computed):",
                    }
                )
                for img_base64 in images[:half]:
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64,
                            },
                        }
                    )

                # Add label for simulated trajectory frames
                content.append(
                    {
                        "type": "text",
                        "text": "SIMULATED TRAJECTORY (what actually happened in physics simulation):",
                    }
                )
                for img_base64 in images[half:]:
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64,
                            },
                        }
                    )
            else:
                # If only a few images, just add them without labels
                for img_base64 in images:
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64,
                            },
                        }
                    )

        # Add the main text message
        content.append({"type": "text", "text": full_user_message})

        try:
            # Use streaming to handle long requests
            response_text = ""
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
            ) as stream:
                for text in stream.text_stream:
                    response_text += text

            return response_text

        except Exception as e:
            print(f"Error calling Claude API with vision: {e}")
            raise

    def extract_code_blocks(self, response: str) -> list[str]:
        """
        Extract Python code blocks from LLM response.

        Args:
            response: Raw LLM response text

        Returns:
            List of Python code blocks found in the response
        """
        code_blocks = []
        lines = response.split("\n")
        in_code_block = False
        current_block: list[str] = []

        for line in lines:
            if line.strip().startswith("```python") or line.strip() == "```python":
                in_code_block = True
                current_block = []
            elif line.strip() == "```" and in_code_block:
                in_code_block = False
                if current_block:
                    code_blocks.append("\n".join(current_block))
                current_block = []
            elif in_code_block:
                current_block.append(line)

        return code_blocks

    def extract_raw_code(self, response: str) -> str:
        """
        Enhanced code extraction from LLM response with better heuristics for MPC configuration.

        This function extracts complete MPC configuration code including:
        - mpc.set_* calls (task_name, duration, time_step, contact_sequence)
        - Variable assignments (phases, contact_seq, etc.)
        - Function definitions (constraint functions)
        - mpc.add_constraint() calls

        Args:
            response: Raw LLM response text

        Returns:
            Extracted Python code
        """
        # First try to extract from code blocks
        code_blocks = self.extract_code_blocks(response)
        if code_blocks:
            # Return the longest code block (likely the complete MPC configuration)
            longest_block = max(code_blocks, key=len)
            return longest_block

        # If no code blocks, use enhanced raw extraction for MPC configuration
        cleaned = response.strip()
        lines = cleaned.split("\n")

        # MPC configuration patterns to look for
        mpc_patterns = [
            "mpc.set_",
            "mpc.add_",
            "mpc._create_",
            "phases =",
            "phases=",
            "contact_seq",
            "contact_sequence",
        ]

        # Find the start of MPC configuration code
        start_idx = None
        end_idx = len(lines)

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Look for MPC configuration calls first (these come before def)
            if any(pattern in stripped for pattern in mpc_patterns):
                start_idx = i
                break

            # Also look for function definitions as fallback
            if stripped.startswith("def "):
                start_idx = i
                break

        # If we found MPC config, look backwards for any preceding variable assignments
        if start_idx is not None:
            # Check if there are relevant variable assignments just before
            while start_idx > 0:
                prev_line = lines[start_idx - 1].strip()
                # Include comment lines and variable assignments that might be part of config
                if (
                    prev_line.startswith("#")
                    or "=" in prev_line
                    or prev_line.startswith("def ")
                    or any(pattern in prev_line for pattern in mpc_patterns)
                ):
                    # But skip obvious non-code explanatory text
                    if any(
                        prev_line.lower().startswith(skip)
                        for skip in ["here", "the ", "this ", "note:", "i'll", "let me"]
                    ):
                        break
                    start_idx -= 1
                else:
                    break

        # If we found a starting point, extract from there
        if start_idx is not None:
            code_lines = lines[start_idx:end_idx]

            # Clean up common suffixes
            while code_lines and not code_lines[-1].strip():
                code_lines.pop()  # Remove empty lines at end

            # Remove common explanatory text at the end
            while code_lines:
                last_line = code_lines[-1].strip().lower()
                if (
                    last_line.startswith("#")
                    and ("explanation" in last_line or "note" in last_line)
                ) or any(
                    phrase in last_line
                    for phrase in [
                        "this function",
                        "this code",
                        "this will",
                        "this should",
                    ]
                ):
                    code_lines.pop()
                else:
                    break

            return "\n".join(code_lines)

        # Fallback: extract any Python-like content that looks like MPC configuration
        python_lines = []
        in_code_section = False

        for line in lines:
            stripped = line.strip()

            # Skip obvious non-code lines
            if any(
                stripped.lower().startswith(skip)
                for skip in [
                    "here",
                    "the ",
                    "this ",
                    "note:",
                    "i'll",
                    "let me",
                    "```",
                ]
            ):
                if in_code_section and stripped.startswith("```"):
                    break  # End of code block
                continue

            # Check if line looks like Python/MPC configuration code
            is_code_line = (
                stripped.startswith("def ")
                or stripped.startswith("return ")
                or stripped.startswith("if ")
                or stripped.startswith("for ")
                or stripped.startswith("while ")
                or stripped.startswith("mpc.")
                or "=" in stripped
                or stripped.startswith("    ")  # Indented line
                or stripped.startswith("#")  # Comments within code
            )

            if is_code_line:
                in_code_section = True
                python_lines.append(line)
            elif in_code_section and not stripped:
                python_lines.append(line)  # Keep empty lines within code
            elif in_code_section and stripped:
                # Check if this might be a continuation of code
                # (e.g., after a blank line in function body)
                if stripped.startswith("mpc.") or stripped.startswith("def "):
                    python_lines.append(line)
                else:
                    # Non-Python line after code started - might be end of code
                    break

        return "\n".join(python_lines) if python_lines else cleaned
