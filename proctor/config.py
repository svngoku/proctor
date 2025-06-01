"""
Configuration for LLM services using litellm with openrouter.
"""

import os
from typing import Dict, Any

# Default configuration for the LLM
DEFAULT_LLM_CONFIG = {
    "model": "openai/gpt-4o",  # Default model to use
    "api_base": "https://openrouter.ai/api/v1",
    "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
    "max_tokens": 1000,
    "temperature": 0.7,
}


def get_llm_config() -> Dict[str, Any]:
    """
    Get the LLM configuration with environment variables if available.
    Returns:
        Dict[str, Any]: The LLM configuration
    """
    config = DEFAULT_LLM_CONFIG.copy()

    # Override with environment variables if present
    if api_key := os.environ.get("OPENROUTER_API_KEY"):
        config["api_key"] = api_key

    if model := os.environ.get("OPENROUTER_MODEL"):
        config["model"] = model

    return config
