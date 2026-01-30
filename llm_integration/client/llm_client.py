"""LLM client for constraint generation using Anthropic's Claude."""

from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from ..logging_config import logger
from .code_extraction import extract_code_blocks, extract_raw_code


class LLMClient:
    """Client for interacting with Claude API to generate optimization constraints."""

    # Assign imported functions directly as methods (no wrapper needed)
    extract_code_blocks = staticmethod(extract_code_blocks)
    extract_raw_code = staticmethod(extract_raw_code)

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
            logger.error(f"Error calling Claude API: {e}")
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
                messages=[{"role": "user", "content": content}],  # type: ignore[typeddict-item]
            ) as stream:
                for text in stream.text_stream:
                    response_text += text

            return response_text

        except Exception as e:
            logger.error(f"Error calling Claude API with vision: {e}")
            raise
