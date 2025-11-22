"""LLM client for constraint generation using Anthropic's Claude."""

import os
from typing import List, Optional

from anthropic import Anthropic
from dotenv import load_dotenv


class LLMClient:
    """Client for interacting with Claude API to generate optimization constraints."""

    def __init__(self) -> None:
        """Initialize the LLM client with environment configuration."""
        load_dotenv()

        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-4-sonnet-20250514")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4000"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment variables. "
                "Please set your API key in the .env file."
            )

        self.client = Anthropic(api_key=self.api_key)

    def generate_constraints(
        self, system_prompt: str, user_message: str, context: Optional[str] = None
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
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": full_user_message}],
            )

            # Extract text from response, handling different content types
            content_block = response.content[0]
            if hasattr(content_block, "text"):
                return content_block.text  # type: ignore
            else:
                return str(content_block)

        except Exception as e:
            print(f"Error calling Claude API: {e}")
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
        current_block: List[str] = []

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
        Extract code from LLM response, trying code blocks first, then raw content.

        Args:
            response: Raw LLM response text

        Returns:
            Extracted Python code
        """
        # First try to extract from code blocks
        code_blocks = self.extract_code_blocks(response)
        if code_blocks:
            return code_blocks[0]

        # If no code blocks, treat entire response as code (for cases where LLM outputs raw code)
        # Clean up common non-code prefixes/suffixes
        cleaned = response.strip()

        # Remove common explanatory text
        lines = cleaned.split("\n")
        start_idx = 0
        end_idx = len(lines)

        # Find first line that looks like Python code
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (
                stripped.startswith("def ")
                or stripped.startswith("class ")
                or stripped.startswith("import ")
            ):
                start_idx = i
                break

        # Take everything from the first def/class/import to the end
        code_lines = lines[start_idx:end_idx]

        return "\n".join(code_lines)
