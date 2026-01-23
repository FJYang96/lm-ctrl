"""LLM client for constraint generation using Anthropic's Claude."""

import os

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
        Enhanced code extraction from LLM response with better heuristics for repair iterations.

        Args:
            response: Raw LLM response text

        Returns:
            Extracted Python code
        """
        # First try to extract from code blocks
        code_blocks = self.extract_code_blocks(response)
        if code_blocks:
            # Return the longest code block (likely the main function)
            longest_block = max(code_blocks, key=len)
            return longest_block

        # If no code blocks, use enhanced raw extraction
        cleaned = response.strip()
        lines = cleaned.split("\n")

        # Enhanced code detection - look for function definitions
        start_idx = None
        end_idx = len(lines)

        # Find first line that looks like Python code
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Look for function definitions, not imports (since we don't allow imports)
            if stripped.startswith("def "):
                start_idx = i
                break
            # Also check for constraint-specific function names
            if any(
                name in stripped
                for name in ["constraint", "jump", "backflip", "spin", "hop"]
            ) and stripped.startswith("def "):
                start_idx = i
                break

        # If we found a starting point, extract from there
        if start_idx is not None:
            code_lines = lines[start_idx:end_idx]

            # Clean up common suffixes
            while code_lines and not code_lines[-1].strip():
                code_lines.pop()  # Remove empty lines at end

            # Remove common explanatory text at the end
            while code_lines:
                last_line = code_lines[-1].strip()
                if (
                    last_line.startswith("#")
                    or "explanation" in last_line.lower()
                    or "note:" in last_line.lower()
                    or "this function" in last_line.lower()
                ):
                    code_lines.pop()
                else:
                    break

            return "\n".join(code_lines)

        # Fallback: if no function found, try to extract any Python-like content
        python_lines = []
        in_code_section = False

        for line in lines:
            stripped = line.strip()

            # Skip obvious non-code lines
            if (
                stripped.startswith("Here")
                or stripped.startswith("The")
                or stripped.startswith("This")
                or stripped.startswith("Note")
                or stripped.startswith("```")
            ):
                continue

            # Check if line looks like Python code
            if (
                stripped.startswith("def ")
                or stripped.startswith("return ")
                or stripped.startswith("if ")
                or stripped.startswith("for ")
                or stripped.startswith("while ")
                or "=" in stripped
                or stripped.startswith("    ")
            ):  # Indented line
                in_code_section = True
                python_lines.append(line)
            elif in_code_section and not stripped:
                python_lines.append(line)  # Keep empty lines within code
            elif in_code_section and stripped:
                # Non-Python line after code started - might be end of code
                break

        return "\n".join(python_lines) if python_lines else cleaned
