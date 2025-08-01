"""
Utility functions for prompt techniques.
"""

import textwrap
import logging
import asyncio
import time
from typing import Dict, Any, Optional, Union, AsyncIterator, Iterator
import litellm
from litellm.exceptions import (
    RateLimitError,
    BadRequestError,
    AuthenticationError,
    ContextWindowExceededError,
    ContentPolicyViolationError,
    ServiceUnavailableError,
    OpenAIError,
    Timeout,
)
from rich.logging import RichHandler
from .config import get_llm_config

# --- Logger Setup ---
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)

log = logging.getLogger("rich")
# --- End Logger Setup ---


def dedent_prompt(prompt: str) -> str:
    """
    Remove common leading whitespace from a multi-line prompt string.

    Args:
        prompt (str): The prompt string to dedent

    Returns:
        str: The dedented prompt
    """
    return textwrap.dedent(prompt).strip()


class LLMError(Exception):
    """Exception raised for errors in LLM API calls."""

    pass


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None,
    max_retries: int = 2,
) -> str:
    """
    Call the LLM with the given prompt using litellm with openrouter.

    Args:
        prompt (str): The user prompt to send
        system_prompt (Optional[str]): Optional system prompt to use
        config_override (Optional[Dict[str, Any]]): Override default config values
        max_retries (int): Maximum number of retry attempts for transient errors

    Returns:
        str: The LLM response content

    Raises:
        LLMError: If there are persistent issues with the LLM call after retries
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt must be a non-empty string")

    config = get_llm_config()

    # Apply config overrides if provided
    if config_override:
        config.update(config_override)

    # Validate required configuration
    if not config.get("api_key"):
        log.error("Missing API key in configuration")
        raise LLMError(
            "Missing API key. Please set OPENROUTER_API_KEY environment variable."
        )

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    log.info("Attempting to call LLM...")
    log.debug(f"LLM Config: {config}")
    log.debug(f"Messages: {messages}")

    # Track retry attempts
    attempts = 0
    last_error = None

    while attempts <= max_retries:
        try:
            # For OpenRouter, we need to set custom_llm_provider
            if "openrouter.ai" in config.get("api_base", ""):
                response = litellm.completion(
                    model=config["model"],
                    messages=messages,
                    api_base=config["api_base"],
                    api_key=config["api_key"],
                    max_tokens=config.get("max_tokens", 1000),
                    temperature=config.get("temperature", 0.7),
                    custom_llm_provider="openrouter",
                )
            else:
                response = litellm.completion(
                    model=config["model"],
                    messages=messages,
                    api_base=config["api_base"],
                    api_key=config["api_key"],
                    max_tokens=config.get("max_tokens", 1000),
                    temperature=config.get("temperature", 0.7),
                )

            log.debug(f"Raw LLM Response object: {response}")

            # Process response
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                log.info("LLM call successful.")
                return content
            else:
                log.error("Received unexpected response format from LLM.")
                log.error(f"Response object: {response}")
                raise LLMError("Unexpected response format from LLM")

        except (
            litellm.exceptions.RateLimitError,
            litellm.exceptions.BadRequestError,
            litellm.exceptions.OpenAIError,
            litellm.exceptions.ServiceUnavailableError,
        ) as e:
            # These are potentially retryable errors
            last_error = e
            attempts += 1

            if attempts <= max_retries:
                retry_delay = 2**attempts  # Exponential backoff
                log.warning(
                    f"Retryable error: {str(e)}. Retrying in {retry_delay}s... (Attempt {attempts}/{max_retries})"
                )
                import time

                time.sleep(retry_delay)
            else:
                # Max retries exceeded
                break

        except Exception as e:
            # Non-retryable error
            log.exception(f"Non-retryable error calling LLM: {e}")
            raise LLMError(f"Error calling LLM: {str(e)}")

    # If we've exhausted retries
    if last_error:
        log.error(f"Failed after {max_retries} retries: {str(last_error)}")
        raise LLMError(f"Error after {max_retries} retries: {str(last_error)}")

    # Fallback error (should not reach here)
    raise LLMError("Unknown error occurred when calling LLM")

async def call_llm_async(
    prompt: str,
    system_prompt: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None,
    max_retries: int = 2,
) -> str:
    """
    Asynchronous version of call_llm using litellm.acompletion.

    Args:
        prompt (str): The user prompt to send
        system_prompt (Optional[str]): Optional system prompt to use
        config_override (Optional[Dict[str, Any]]): Override default config values
        max_retries (int): Maximum number of retry attempts for transient errors

    Returns:
        str: The LLM response content

    Raises:
        LLMError: If there are persistent issues with the LLM call after retries
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt must be a non-empty string")

    config = get_llm_config()

    # Apply config overrides if provided
    if config_override:
        config.update(config_override)

    # Validate required configuration
    if not config.get("api_key"):
        log.error("Missing API key in configuration")
        raise LLMError(
            "Missing API key. Please set OPENROUTER_API_KEY environment variable."
        )

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    log.info("Attempting to call LLM asynchronously...")
    log.debug(f"LLM Config: {config}")
    log.debug(f"Messages: {messages}")

    # Track retry attempts
    attempts = 0
    last_error = None

    while attempts <= max_retries:
        try:
            # For OpenRouter, we need to set custom_llm_provider
            if "openrouter.ai" in config.get("api_base", ""):
                response = await litellm.acompletion(
                    model=config["model"],
                    messages=messages,
                    api_base=config["api_base"],
                    api_key=config["api_key"],
                    max_tokens=config.get("max_tokens", 1000),
                    temperature=config.get("temperature", 0.7),
                    custom_llm_provider="openrouter",
                )
            else:
                response = await litellm.acompletion(
                    model=config["model"],
                    messages=messages,
                    api_base=config["api_base"],
                    api_key=config["api_key"],
                    max_tokens=config.get("max_tokens", 1000),
                    temperature=config.get("temperature", 0.7),
                )

            log.debug(f"Raw LLM Response object: {response}")

            # Process response
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                log.info("LLM call successful.")
                return content
            else:
                log.error("Received unexpected response format from LLM.")
                log.error(f"Response object: {response}")
                raise LLMError("Unexpected response format from LLM")

        except (
            RateLimitError,
            BadRequestError,
            OpenAIError,
            ServiceUnavailableError,
            Timeout,
        ) as e:
            # These are potentially retryable errors
            last_error = e
            attempts += 1

            if attempts <= max_retries:
                retry_delay = 2**attempts  # Exponential backoff
                log.warning(
                    f"Retryable error: {str(e)}. Retrying in {retry_delay}s... (Attempt {attempts}/{max_retries})"
                )
                await asyncio.sleep(retry_delay)
            else:
                # Max retries exceeded
                break

        except Exception as e:
            # Non-retryable error
            log.exception(f"Non-retryable error calling LLM: {e}")
            raise LLMError(f"Error calling LLM: {str(e)}")

    # If we've exhausted retries
    if last_error:
        log.error(f"Failed after {max_retries} retries: {str(last_error)}")
        raise LLMError(f"Error after {max_retries} retries: {str(last_error)}")

    # Fallback error (should not reach here)
    raise LLMError("Unknown error occurred when calling LLM")


def call_llm_stream(
    prompt: str,
    system_prompt: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None,
) -> Iterator[str]:
    """
    Call the LLM with streaming response.

    Args:
        prompt (str): The user prompt to send
        system_prompt (Optional[str]): Optional system prompt to use
        config_override (Optional[Dict[str, Any]]): Override default config values

    Yields:
        str: Chunks of the LLM response content

    Raises:
        LLMError: If there are issues with the LLM call
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt must be a non-empty string")

    config = get_llm_config()

    # Apply config overrides if provided
    if config_override:
        config.update(config_override)

    # Validate required configuration
    if not config.get("api_key"):
        log.error("Missing API key in configuration")
        raise LLMError(
            "Missing API key. Please set OPENROUTER_API_KEY environment variable."
        )

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    log.info("Attempting to call LLM with streaming...")
    log.debug(f"LLM Config: {config}")
    log.debug(f"Messages: {messages}")

    try:
        # For OpenRouter, we need to set custom_llm_provider
        if "openrouter.ai" in config.get("api_base", ""):
            response = litellm.completion(
                model=config["model"],
                messages=messages,
                api_base=config["api_base"],
                api_key=config["api_key"],
                max_tokens=config.get("max_tokens", 1000),
                temperature=config.get("temperature", 0.7),
                stream=True,
                custom_llm_provider="openrouter",
            )
        else:
            response = litellm.completion(
                model=config["model"],
                messages=messages,
                api_base=config["api_base"],
                api_key=config["api_key"],
                max_tokens=config.get("max_tokens", 1000),
                temperature=config.get("temperature", 0.7),
                stream=True,
            )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        log.exception(f"Error during streaming LLM call: {e}")
        raise LLMError(f"Error during streaming: {str(e)}")


async def call_llm_async_stream(
    prompt: str,
    system_prompt: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[str]:
    """
    Asynchronously call the LLM with streaming response.

    Args:
        prompt (str): The user prompt to send
        system_prompt (Optional[str]): Optional system prompt to use
        config_override (Optional[Dict[str, Any]]): Override default config values

    Yields:
        str: Chunks of the LLM response content

    Raises:
        LLMError: If there are issues with the LLM call
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt must be a non-empty string")

    config = get_llm_config()

    # Apply config overrides if provided
    if config_override:
        config.update(config_override)

    # Validate required configuration
    if not config.get("api_key"):
        log.error("Missing API key in configuration")
        raise LLMError(
            "Missing API key. Please set OPENROUTER_API_KEY environment variable."
        )

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    log.info("Attempting to call LLM asynchronously with streaming...")
    log.debug(f"LLM Config: {config}")
    log.debug(f"Messages: {messages}")

    try:
        # For OpenRouter, we need to set custom_llm_provider
        if "openrouter.ai" in config.get("api_base", ""):
            response = await litellm.acompletion(
                model=config["model"],
                messages=messages,
                api_base=config["api_base"],
                api_key=config["api_key"],
                max_tokens=config.get("max_tokens", 1000),
                temperature=config.get("temperature", 0.7),
                stream=True,
                custom_llm_provider="openrouter",
            )
        else:
            response = await litellm.acompletion(
                model=config["model"],
                messages=messages,
                api_base=config["api_base"],
                api_key=config["api_key"],
                max_tokens=config.get("max_tokens", 1000),
                temperature=config.get("temperature", 0.7),
                stream=True,
            )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        log.exception(f"Error during async streaming LLM call: {e}")
        raise LLMError(f"Error during async streaming: {str(e)}")
