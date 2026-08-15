"""
OpenRouter model registry.

Curated map of recent OpenRouter model identifiers (current as of 2026-06),
plus helpers to look them up. Use the bare provider/model id with LiteLLM and
let it detect OpenRouter from ``api_base`` (do NOT prefix with ``openrouter/``).
"""

from typing import Dict, List, Optional

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Recent general-purpose models grouped by provider (newest first).
RECENT_MODELS: Dict[str, List[str]] = {
    "openai": [
        "openai/gpt-5.5-pro",
        "openai/gpt-5.5",
        "openai/gpt-5.4",
        "openai/gpt-5.4-mini",
        "openai/gpt-5.4-nano",
        "openai/gpt-chat-latest",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
    ],
    "anthropic": [
        "anthropic/claude-opus-4.8",
        "anthropic/claude-opus-4.8-fast",
        "anthropic/claude-opus-4.7",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-opus-4.6",
    ],
    "google": [
        "google/gemini-3.5-flash",
        "google/gemini-3.1-flash-lite",
        "google/gemini-3-pro-image",
        "google/gemma-4-31b-it",
        "google/gemma-4-26b-a4b-it",
    ],
    "deepseek": [
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v3.2",
        "deepseek/deepseek-r1-0528",
    ],
    "meta-llama": [
        "meta-llama/llama-4-maverick",
        "meta-llama/llama-4-scout",
        "meta-llama/llama-3.3-70b-instruct",
    ],
    "mistralai": [
        "mistralai/mistral-large-2512",
        "mistralai/mistral-medium-3-5",
        "mistralai/mistral-small-2603",
        "mistralai/ministral-8b-2512",
    ],
    "qwen": [
        "qwen/qwen3.7-max",
        "qwen/qwen3.7-plus",
        "qwen/qwen3.6-flash",
        "qwen/qwen3.6-max-preview",
    ],
    "x-ai": [
        "x-ai/grok-4.3",
        "x-ai/grok-4.20",
        "x-ai/grok-4.20-multi-agent",
    ],
    "moonshotai": [
        "moonshotai/kimi-k2.7-code",
        "moonshotai/kimi-k2.6",
        "moonshotai/kimi-k2-thinking",
    ],
    "z-ai": [
        "z-ai/glm-5.2",
        "z-ai/glm-5.1",
        "z-ai/glm-5",
    ],
    "nvidia": [
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-super-120b-a12b",
    ],
    "minimax": [
        "minimax/minimax-m3",
        "minimax/minimax-m2.7",
    ],
    "cohere": [
        "cohere/command-a",
        "cohere/command-r-plus-08-2024",
    ],
}

# Recent open-weight / open-source models grouped by family (newest first).
OPEN_SOURCE_MODELS: Dict[str, List[str]] = {
    "moonshotai": [
        "moonshotai/kimi-k2.7-code",
        "moonshotai/kimi-k2.6",
        "moonshotai/kimi-k2-thinking",
        "moonshotai/kimi-k2.5",
    ],
    "z-ai": [
        "z-ai/glm-5.2",
        "z-ai/glm-5.1",
        "z-ai/glm-5",
        "z-ai/glm-4.7",
        "z-ai/glm-4.6",
        "z-ai/glm-4.5-air",
    ],
    "minimax": [
        "minimax/minimax-m3",
        "minimax/minimax-m2.7",
        "minimax/minimax-m2.5",
        "minimax/minimax-m2",
    ],
    "deepseek": [
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v3.2",
        "deepseek/deepseek-r1-0528",
    ],
    "qwen": [
        "qwen/qwen3.6-35b-a3b",
        "qwen/qwen3.6-27b",
        "qwen/qwen3.5-122b-a10b",
        "qwen/qwen3.5-9b",
    ],
    "meta-llama": [
        "meta-llama/llama-4-maverick",
        "meta-llama/llama-4-scout",
        "meta-llama/llama-3.3-70b-instruct",
    ],
    "mistralai": [
        "mistralai/mistral-large-2512",
        "mistralai/devstral-2512",
        "mistralai/codestral-2508",
        "mistralai/ministral-8b-2512",
    ],
    "nvidia": [
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-3-nano-30b-a3b",
    ],
}

# Open-weight, coding-tuned models.
CODING_MODELS: List[str] = [
    "moonshotai/kimi-k2.7-code",
    "z-ai/glm-5.2",
    "deepseek/deepseek-v4-pro",
    "mistralai/devstral-2512",
    "mistralai/codestral-2508",
    "minimax/minimax-m3",
]

# Free-tier (no-cost) endpoints.
FREE_MODELS: List[str] = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

# Floating aliases that always point to a provider's newest model of that tier.
FLOATING_ALIASES: Dict[str, str] = {
    "openai-latest": "~openai/gpt-latest",
    "openai-mini-latest": "~openai/gpt-mini-latest",
    "claude-opus-latest": "~anthropic/claude-opus-latest",
    "claude-sonnet-latest": "~anthropic/claude-sonnet-latest",
    "claude-haiku-latest": "~anthropic/claude-haiku-latest",
    "gemini-pro-latest": "~google/gemini-pro-latest",
    "gemini-flash-latest": "~google/gemini-flash-latest",
    "kimi-latest": "~moonshotai/kimi-latest",
}

# Named constants for popular flagships.
DEFAULT_MODEL = "openai/gpt-4o"
GPT_5_5_PRO = "openai/gpt-5.5-pro"
GPT_5_5 = "openai/gpt-5.5"
GPT_5_4 = "openai/gpt-5.4"
GPT_5_4_MINI = "openai/gpt-5.4-mini"
CLAUDE_OPUS_4_8 = "anthropic/claude-opus-4.8"
CLAUDE_SONNET_4_6 = "anthropic/claude-sonnet-4.6"
GEMINI_3_5_FLASH = "google/gemini-3.5-flash"
DEEPSEEK_V4_PRO = "deepseek/deepseek-v4-pro"
DEEPSEEK_R1 = "deepseek/deepseek-r1-0528"
LLAMA_4_MAVERICK = "meta-llama/llama-4-maverick"
MISTRAL_LARGE = "mistralai/mistral-large-2512"
QWEN_3_7_MAX = "qwen/qwen3.7-max"
GROK_4_3 = "x-ai/grok-4.3"
KIMI_K2_7_CODE = "moonshotai/kimi-k2.7-code"
KIMI_K2_THINKING = "moonshotai/kimi-k2-thinking"
GLM_5_2 = "z-ai/glm-5.2"
GLM_5_1 = "z-ai/glm-5.1"
MINIMAX_M3 = "minimax/minimax-m3"
DEVSTRAL = "mistralai/devstral-2512"
CODESTRAL = "mistralai/codestral-2508"

# Flat list of every curated model id (recent + open-source).
ALL_OPENROUTER_MODELS: List[str] = list(
    dict.fromkeys(
        [model for models in RECENT_MODELS.values() for model in models]
        + [model for models in OPEN_SOURCE_MODELS.values() for model in models]
    )
)


def list_models(provider: Optional[str] = None) -> List[str]:
    """
    List curated recent OpenRouter models, optionally filtered by provider.

    Args:
        provider (Optional[str]): Provider key (e.g., "openai", "anthropic").
            If None, returns all curated models.

    Returns:
        List[str]: Matching model identifiers.
    """
    if provider is None:
        return list(ALL_OPENROUTER_MODELS)
    return list(RECENT_MODELS.get(provider, []))


def list_open_source_models(family: Optional[str] = None) -> List[str]:
    """
    List curated recent open-weight models, optionally filtered by family.

    Args:
        family (Optional[str]): Family key (e.g., "moonshotai", "z-ai", "minimax").
            If None, returns all curated open-source models.

    Returns:
        List[str]: Matching model identifiers.
    """
    if family is None:
        return [m for models in OPEN_SOURCE_MODELS.values() for m in models]
    return list(OPEN_SOURCE_MODELS.get(family, []))


def is_known_model(model: str) -> bool:
    """Return True if ``model`` is in the curated registry or floating aliases."""
    return (
        model in ALL_OPENROUTER_MODELS
        or model in FLOATING_ALIASES
        or model in FLOATING_ALIASES.values()
    )
