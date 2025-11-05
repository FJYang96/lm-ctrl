"""
Configuration loader for LLM integration
Handles environment variables and .env files
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class LLMIntegrationConfig:
    """Configuration for LLM integration system"""

    # LLM Provider settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None

    # Generation parameters
    max_tokens: int = 2000
    temperature: float = 0.1
    timeout: int = 60

    # Iteration control
    max_llm_iterations: int = 10
    trajectory_optimization_timeout: int = 120

    # Logging and output
    log_llm_iterations: bool = True
    save_intermediate_results: bool = True
    results_dir: str = "results/llm_iterations"

    # Mock mode for testing
    use_mock_llm: bool = False


def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from .env file

    Args:
        env_path: Path to .env file

    Returns:
        Dictionary of environment variables
    """
    env_vars: Dict[str, str] = {}

    if not Path(env_path).exists():
        return env_vars

    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # Parse key=value pairs
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")

                    if value and not value.startswith("#"):
                        env_vars[key] = value

    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")

    return env_vars


def get_config() -> LLMIntegrationConfig:
    """
    Load configuration from environment variables and .env file

    Returns:
        LLMIntegrationConfig instance
    """

    # Load .env file first
    env_vars = load_env_file()

    # Update os.environ with .env variables (don't override existing)
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value

    # Determine LLM provider and credentials
    llm_provider = "openai"
    llm_model = "gpt-4"
    llm_api_key = None
    llm_base_url = None

    # Check for OpenAI
    if os.getenv("OPENAI_API_KEY"):
        llm_provider = "openai"
        llm_api_key = os.getenv("OPENAI_API_KEY")
        llm_model = os.getenv("OPENAI_MODEL", "gpt-4")

    # Check for Anthropic
    elif os.getenv("ANTHROPIC_API_KEY"):
        llm_provider = "anthropic"
        llm_api_key = os.getenv("ANTHROPIC_API_KEY")
        llm_model = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")

    # Check for local/custom endpoint
    elif os.getenv("LLM_BASE_URL"):
        llm_provider = "local"
        llm_api_key = os.getenv("LLM_API_KEY", "dummy")
        llm_model = os.getenv("LLM_MODEL", "local-model")
        llm_base_url = os.getenv("LLM_BASE_URL")

    # Helper function to parse boolean values
    def parse_bool(value: Optional[str], default: bool = False) -> bool:
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    # Helper function to parse integer values
    def parse_int(value: Optional[str], default: int) -> int:
        try:
            return int(value) if value else default
        except ValueError:
            return default

    # Helper function to parse float values
    def parse_float(value: Optional[str], default: float) -> float:
        try:
            return float(value) if value else default
        except ValueError:
            return default

    config = LLMIntegrationConfig(
        # LLM Provider
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        # Generation parameters
        max_tokens=parse_int(os.getenv("LLM_MAX_TOKENS"), 2000),
        temperature=parse_float(os.getenv("LLM_TEMPERATURE"), 0.1),
        timeout=parse_int(os.getenv("LLM_TIMEOUT"), 60),
        # Iteration control
        max_llm_iterations=parse_int(os.getenv("MAX_LLM_ITERATIONS"), 10),
        trajectory_optimization_timeout=parse_int(
            os.getenv("TRAJECTORY_OPTIMIZATION_TIMEOUT"), 120
        ),
        # Logging and output
        log_llm_iterations=parse_bool(os.getenv("LOG_LLM_ITERATIONS"), True),
        save_intermediate_results=parse_bool(
            os.getenv("SAVE_INTERMEDIATE_RESULTS"), True
        ),
        results_dir=os.getenv("RESULTS_DIR", "results/llm_iterations"),
        # Mock mode
        use_mock_llm=parse_bool(os.getenv("USE_MOCK_LLM"), False),
    )

    return config


def print_config_status() -> None:
    """Print current configuration status for debugging"""

    config = get_config()

    print("🔧 LLM Integration Configuration:")
    print(f"   Provider: {config.llm_provider}")
    print(f"   Model: {config.llm_model}")
    print(f"   API Key: {'✅ Set' if config.llm_api_key else '❌ Missing'}")

    if config.llm_base_url:
        print(f"   Base URL: {config.llm_base_url}")

    print(f"   Max Tokens: {config.max_tokens}")
    print(f"   Temperature: {config.temperature}")
    print(f"   Max Iterations: {config.max_llm_iterations}")

    if config.use_mock_llm:
        print("   🎭 Mock Mode: Enabled")

    if not config.llm_api_key and not config.use_mock_llm:
        print("\n⚠️  No API key found! Set one of:")
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - LLM_API_KEY (for custom endpoints)")
        print("   Or set USE_MOCK_LLM=true for testing")
        print("\n📋 Copy .env.template to .env and add your API key")


if __name__ == "__main__":
    # Test configuration loading
    print_config_status()
