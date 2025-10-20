"""
DSPy Language Model wrapper for LiteLLM integration.

This module provides a bridge between DSPy and Proctor's LiteLLM-based
LLM calling infrastructure, allowing DSPy techniques to leverage the
existing OpenRouter/LiteLLM setup.
"""

import os
from typing import Any, Dict, List, Optional
import dspy
from .utils import call_llm, call_llm_async, log, LLMError
from .config import get_llm_config


class LiteLLMLanguageModel(dspy.LM):
    """
    DSPy Language Model wrapper for LiteLLM.

    This class bridges DSPy's LM interface with Proctor's existing
    LiteLLM-based infrastructure, enabling DSPy techniques to work
    seamlessly with OpenRouter and other LiteLLM-supported providers.

    Args:
        model (str): Model identifier (e.g., "openai/gpt-4o")
        api_base (Optional[str]): API base URL
        api_key (Optional[str]): API key for the provider
        max_tokens (int): Maximum tokens in response
        temperature (float): Sampling temperature
        **kwargs: Additional parameters for litellm.completion

    Example:
        >>> lm = LiteLLMLanguageModel(
        ...     model="openai/gpt-4o",
        ...     api_base="https://openrouter.ai/api/v1",
        ...     api_key=os.getenv("OPENROUTER_API_KEY")
        ... )
        >>> dspy.settings.configure(lm=lm)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs,
    ):
        """Initialize the LiteLLM language model wrapper."""
        super().__init__(model=model or "openai/gpt-4o")

        # Get default config and override with provided values
        config = get_llm_config()

        self.model_name = model or config.get("model", "openai/gpt-4o")
        self.api_base = api_base or config.get("api_base")
        self.api_key = api_key or config.get("api_key")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.kwargs = kwargs

        # Store as config for call_llm
        self.llm_config = {
            "model": self.model_name,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **kwargs,
        }

        log.info(
            f"[DSPy] Initialized LiteLLM language model: {self.model_name}"
        )

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Call the language model with a prompt or messages.

        Args:
            prompt (Optional[str]): Simple text prompt
            messages (Optional[List[Dict]]): Chat messages in OpenAI format
            **kwargs: Additional parameters

        Returns:
            List[str]: List containing the model's response

        Raises:
            LLMError: If the API call fails
        """
        # Merge configs
        config_override = {**self.llm_config, **kwargs}

        # Handle both prompt and messages formats
        if messages:
            # Extract system and user messages
            system_prompt = None
            user_prompt = ""

            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "system":
                    system_prompt = content
                elif role == "user":
                    user_prompt = content if not user_prompt else f"{user_prompt}\n{content}"
                elif role == "assistant":
                    # For few-shot examples, append to user prompt
                    user_prompt = f"{user_prompt}\nAssistant: {content}"

            prompt = user_prompt
        else:
            system_prompt = None

        try:
            # Use existing call_llm infrastructure
            response = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                config_override=config_override,
                max_retries=2,
            )

            # DSPy expects a list of responses
            return [response]

        except LLMError as e:
            log.error(f"[DSPy] LLM call failed: {str(e)}")
            raise

    async def __call__async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Asynchronously call the language model.

        Args:
            prompt (Optional[str]): Simple text prompt
            messages (Optional[List[Dict]]): Chat messages
            **kwargs: Additional parameters

        Returns:
            List[str]: List containing the model's response
        """
        config_override = {**self.llm_config, **kwargs}

        if messages:
            system_prompt = None
            user_prompt = ""

            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "system":
                    system_prompt = content
                elif role == "user":
                    user_prompt = content if not user_prompt else f"{user_prompt}\n{content}"

            prompt = user_prompt
        else:
            system_prompt = None

        try:
            response = await call_llm_async(
                prompt=prompt,
                system_prompt=system_prompt,
                config_override=config_override,
                max_retries=2,
            )

            return [response]

        except LLMError as e:
            log.error(f"[DSPy] Async LLM call failed: {str(e)}")
            raise

    def inspect_history(self, n: int = 1) -> List[Dict[str, Any]]:
        """
        Inspect recent history (for DSPy compatibility).

        Args:
            n (int): Number of recent calls to inspect

        Returns:
            List[Dict]: History entries
        """
        # Note: Full history tracking would require additional state
        # For now, return empty list (can be enhanced later)
        return []


def configure_dspy_with_litellm(
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> LiteLLMLanguageModel:
    """
    Configure DSPy to use LiteLLM as the language model backend.

    This is a convenience function that creates a LiteLLMLanguageModel
    and configures it as the default DSPy LM.

    Args:
        model (Optional[str]): Model identifier
        api_base (Optional[str]): API base URL
        api_key (Optional[str]): API key
        **kwargs: Additional LM parameters

    Returns:
        LiteLLMLanguageModel: The configured language model

    Example:
        >>> from proctor.dspy_lm import configure_dspy_with_litellm
        >>> lm = configure_dspy_with_litellm()
        >>> # Now all DSPy techniques will use LiteLLM
    """
    lm = LiteLLMLanguageModel(
        model=model,
        api_base=api_base,
        api_key=api_key,
        **kwargs,
    )

    # Configure DSPy to use this LM
    dspy.settings.configure(lm=lm)

    log.info("[DSPy] Configured DSPy with LiteLLM backend")

    return lm
