"""
LLM client for generating constraint functions
Supports multiple LLM providers with API key configuration
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class LLMConfig:
    """Configuration for LLM client"""

    provider: str = "openai"  # openai, anthropic, local
    model: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.1
    timeout: int = 60


class LLMClient:
    """
    Universal LLM client supporting multiple providers
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM client

        Args:
            config: LLM configuration, defaults from environment
        """
        self.config = config or self._load_config_from_env()
        self._validate_config()

    def _load_config_from_env(self) -> LLMConfig:
        """Load configuration from environment variables"""

        # Determine provider based on available API keys
        provider = "openai"  # default
        api_key = None
        model = "gpt-4"

        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("OPENAI_MODEL", "gpt-4")

        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
            api_key = os.getenv("ANTHROPIC_API_KEY")
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")

        elif os.getenv("LLM_BASE_URL"):  # Local/custom endpoint
            provider = "local"
            api_key = os.getenv("LLM_API_KEY", "dummy")
            model = os.getenv("LLM_MODEL", "local-model")

        return LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            timeout=int(os.getenv("LLM_TIMEOUT", "60")),
        )

    def _validate_config(self) -> None:
        """Validate LLM configuration"""
        if not self.config.api_key:
            raise ValueError(
                "No API key found. Please set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, or LLM_API_KEY"
            )

        if self.config.provider not in ["openai", "anthropic", "local"]:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response from LLM

        Args:
            prompt: Input prompt text
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated response text
        """

        # Override config with kwargs
        config = LLMConfig(
            provider=self.config.provider,
            model=kwargs.get("model", self.config.model),
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            timeout=kwargs.get("timeout", self.config.timeout),
        )

        print(f"🧠 Calling {config.provider} LLM: {config.model}")

        if config.provider == "openai":
            return self._call_openai(prompt, config)
        elif config.provider == "anthropic":
            return self._call_anthropic(prompt, config)
        elif config.provider == "local":
            return self._call_local(prompt, config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    def _call_openai(self, prompt: str, config: LLMConfig) -> str:
        """Call OpenAI API"""
        try:
            import openai

            client = openai.OpenAI(api_key=config.api_key)

            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                timeout=config.timeout,
            )

            return response.choices[0].message.content

        except ImportError:
            # Fallback to direct API call
            return self._call_openai_direct(prompt, config)
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    def _call_openai_direct(self, prompt: str, config: LLMConfig) -> str:
        """Direct OpenAI API call using requests"""

        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=config.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenAI API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def _call_anthropic(self, prompt: str, config: LLMConfig) -> str:
        """Call Anthropic Claude API"""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=config.api_key)

            response = client.messages.create(
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text

        except ImportError:
            # Fallback to direct API call
            return self._call_anthropic_direct(prompt, config)
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")

    def _call_anthropic_direct(self, prompt: str, config: LLMConfig) -> str:
        """Direct Anthropic API call using requests"""

        headers = {
            "x-api-key": config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=config.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Anthropic API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        return result["content"][0]["text"]

    def _call_local(self, prompt: str, config: LLMConfig) -> str:
        """Call local/custom LLM endpoint"""

        if not config.base_url:
            raise ValueError("base_url required for local provider")

        headers = {"Content-Type": "application/json"}

        if config.api_key and config.api_key != "dummy":
            headers["Authorization"] = f"Bearer {config.api_key}"

        # Try OpenAI-compatible format first
        data = {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }

        try:
            response = requests.post(
                f"{config.base_url.rstrip('/')}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=config.timeout,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]

        except Exception:
            pass

        # Fallback to simple prompt format
        data = {
            "prompt": prompt,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }

        response = requests.post(
            f"{config.base_url.rstrip('/')}/generate",
            headers=headers,
            json=data,
            timeout=config.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Local LLM error: {response.status_code} - {response.text}"
            )

        result = response.json()

        # Handle different response formats
        if "text" in result:
            return result["text"]
        elif "response" in result:
            return result["response"]
        elif "output" in result:
            return result["output"]
        else:
            raise RuntimeError(f"Unknown response format: {result}")


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing without API calls
    """

    def __init__(self):
        """Initialize mock client"""
        self.config = LLMConfig(provider="mock", model="mock-model", api_key="mock-key")

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate mock response"""

        print("🎭 Using mock LLM client")

        # Simple mock constraint function for backflip
        mock_response = '''
def generated_constraints(x_k, u_k, kinodynamic_model, config, contact_k, k, horizon):
    """
    Generated constraints for backflip maneuver
    """

    # Terminal rotation constraint - ensure full backflip
    if k == horizon:
        initial_pitch = 0.0  # Assume starting from level
        target_rotation = initial_pitch + 2 * np.pi  # Full backward rotation
        pitch_constraint = x_k[MP_X_BASE_EUL][1] - target_rotation
        return pitch_constraint, -0.1, 0.1  # ±0.1 rad tolerance

    # Height clearance during flight
    if k > 10 and k < horizon - 10:  # Middle portion of trajectory
        min_height = 0.5  # Minimum 50cm height
        height_constraint = x_k[MP_X_BASE_POS][2] - min_height
        return height_constraint, 0.0, cs.inf

    # No additional constraints for other time steps
    return None
'''

        time.sleep(0.5)  # Simulate API delay
        return mock_response

    def _validate_config(self) -> None:
        """Skip validation for mock client"""
        pass


# Convenience function to get configured client
def get_llm_client(use_mock: bool = False) -> LLMClient:
    """
    Get configured LLM client

    Args:
        use_mock: Use mock client instead of real API

    Returns:
        Configured LLM client
    """

    if use_mock:
        return MockLLMClient()

    try:
        return LLMClient()
    except ValueError as e:
        print(f"⚠️  LLM configuration error: {e}")
        print("🎭 Falling back to mock client")
        return MockLLMClient()
