"""
Utility functions for prompt techniques.
"""

import time
import textwrap
import logging
import asyncio
from typing import Dict, Any, Optional, AsyncIterator, Iterator

from openrouter import OpenRouter
from openrouter.errors import (
    OpenRouterError,
    BadRequestResponseError,
    UnauthorizedResponseError,
    ForbiddenResponseError,
    NotFoundResponseError,
    PaymentRequiredResponseError,
    UnprocessableEntityResponseError,
    ConflictResponseError,
    PayloadTooLargeResponseError,
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

# Client (4xx) errors that should never be retried.
NON_RETRYABLE_ERRORS = (
    BadRequestResponseError,
    UnauthorizedResponseError,
    ForbiddenResponseError,
    NotFoundResponseError,
    PaymentRequiredResponseError,
    UnprocessableEntityResponseError,
    ConflictResponseError,
    PayloadTooLargeResponseError,
)


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


def _normalize_model(model: str) -> str:
    """Strip a leading ``openrouter/`` prefix; the SDK targets OpenRouter natively."""
    if model.startswith("openrouter/"):
        return model[len("openrouter/") :]
    return model


def _prepare_config(
    config_override: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge overrides into the base config and validate the API key."""
    config = get_llm_config()
    if config_override:
        config.update(config_override)

    if not config.get("api_key"):
        log.error("Missing API key in configuration")
        raise LLMError(
            "Missing API key. Please set OPENROUTER_API_KEY environment variable."
        )
    return config


def _build_messages(prompt: str, system_prompt: Optional[str]) -> list:
    """Build the OpenRouter chat messages list."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _openrouter_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"api_key": config["api_key"]}
    if config.get("api_base"):
        kwargs["server_url"] = config["api_base"]
    return kwargs


def _send_kwargs(
    config: Dict[str, Any], messages: list, stream: bool
) -> Dict[str, Any]:
    """Assemble keyword arguments for ``client.chat.send``."""
    kwargs: Dict[str, Any] = {
        "model": _normalize_model(config["model"]),
        "messages": messages,
        "max_tokens": config.get("max_tokens", 1000),
        "temperature": config.get("temperature", 0.7),
        "timeout_ms": int(config.get("timeout", 120) * 1000),
    }
    if stream:
        kwargs["stream"] = True
    return kwargs


def _extract_content(response: Any) -> str:
    """Extract the message content from a non-streaming chat response."""
    choices = getattr(response, "choices", None)
    if choices and choices[0].message and choices[0].message.content is not None:
        return choices[0].message.content
    log.error("Received unexpected response format from LLM.")
    log.error(f"Response object: {response}")
    raise LLMError("Unexpected response format from LLM")


def _chunk_content(event: Any) -> Optional[str]:
    """Extract incremental content from a streaming chat event, if present."""
    choices = getattr(event, "choices", None)
    if choices and getattr(choices[0], "delta", None):
        return choices[0].delta.content
    return None


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None,
    max_retries: int = 2,
) -> str:
    """
    Call the LLM with the given prompt using the OpenRouter SDK.

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

    config = _prepare_config(config_override)
    messages = _build_messages(prompt, system_prompt)

    log.info("Attempting to call LLM...")
    log.debug(f"LLM Config: {config}")
    log.debug(f"Messages: {messages}")

    attempts = 0
    last_error = None

    with OpenRouter(**_openrouter_kwargs(config)) as client:
        while attempts <= max_retries:
            try:
                response = client.chat.send(**_send_kwargs(config, messages, False))
                log.debug(f"Raw LLM Response object: {response}")
                content = _extract_content(response)
                log.info("LLM call successful.")
                return content

            except OpenRouterError as e:
                if isinstance(e, NON_RETRYABLE_ERRORS):
                    log.exception(f"Non-retryable error calling LLM: {e}")
                    raise LLMError(f"Error calling LLM: {str(e)}")

                last_error = e
                attempts += 1
                if attempts <= max_retries:
                    retry_delay = 2**attempts
                    log.warning(
                        f"Retryable error: {str(e)}. Retrying in {retry_delay}s... "
                        f"(Attempt {attempts}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                else:
                    break

            except LLMError:
                raise

            except Exception as e:
                log.exception(f"Non-retryable error calling LLM: {e}")
                raise LLMError(f"Error calling LLM: {str(e)}")

    if last_error:
        log.error(f"Failed after {max_retries} retries: {str(last_error)}")
        raise LLMError(f"Error after {max_retries} retries: {str(last_error)}")

    raise LLMError("Unknown error occurred when calling LLM")


async def call_llm_async(
    prompt: str,
    system_prompt: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None,
    max_retries: int = 2,
) -> str:
    """
    Asynchronous version of call_llm using the OpenRouter SDK.

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

    config = _prepare_config(config_override)
    messages = _build_messages(prompt, system_prompt)

    log.info("Attempting to call LLM asynchronously...")
    log.debug(f"LLM Config: {config}")
    log.debug(f"Messages: {messages}")

    attempts = 0
    last_error = None

    async with OpenRouter(**_openrouter_kwargs(config)) as client:
        while attempts <= max_retries:
            try:
                response = await client.chat.send_async(
                    **_send_kwargs(config, messages, False)
                )
                log.debug(f"Raw LLM Response object: {response}")
                content = _extract_content(response)
                log.info("LLM call successful.")
                return content

            except OpenRouterError as e:
                if isinstance(e, NON_RETRYABLE_ERRORS):
                    log.exception(f"Non-retryable error calling LLM: {e}")
                    raise LLMError(f"Error calling LLM: {str(e)}")

                last_error = e
                attempts += 1
                if attempts <= max_retries:
                    retry_delay = 2**attempts
                    log.warning(
                        f"Retryable error: {str(e)}. Retrying in {retry_delay}s... "
                        f"(Attempt {attempts}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    break

            except LLMError:
                raise

            except Exception as e:
                log.exception(f"Non-retryable error calling LLM: {e}")
                raise LLMError(f"Error calling LLM: {str(e)}")

    if last_error:
        log.error(f"Failed after {max_retries} retries: {str(last_error)}")
        raise LLMError(f"Error after {max_retries} retries: {str(last_error)}")

    raise LLMError("Unknown error occurred when calling LLM")


def call_llm_stream(
    prompt: str,
    system_prompt: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None,
) -> Iterator[str]:
    """
    Call the LLM with a streaming response using the OpenRouter SDK.

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

    config = _prepare_config(config_override)
    messages = _build_messages(prompt, system_prompt)

    log.info("Attempting to call LLM with streaming...")
    log.debug(f"LLM Config: {config}")
    log.debug(f"Messages: {messages}")

    try:
        with OpenRouter(**_openrouter_kwargs(config)) as client:
            stream = client.chat.send(**_send_kwargs(config, messages, True))
            for event in stream:
                content = _chunk_content(event)
                if content:
                    yield content
    except LLMError:
        raise
    except Exception as e:
        log.exception(f"Error during streaming LLM call: {e}")
        raise LLMError(f"Error during streaming: {str(e)}")


async def call_llm_async_stream(
    prompt: str,
    system_prompt: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[str]:
    """
    Asynchronously call the LLM with a streaming response using the OpenRouter SDK.

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

    config = _prepare_config(config_override)
    messages = _build_messages(prompt, system_prompt)

    log.info("Attempting to call LLM asynchronously with streaming...")
    log.debug(f"LLM Config: {config}")
    log.debug(f"Messages: {messages}")

    try:
        async with OpenRouter(**_openrouter_kwargs(config)) as client:
            stream = await client.chat.send_async(
                **_send_kwargs(config, messages, True)
            )
            async for event in stream:
                content = _chunk_content(event)
                if content:
                    yield content
    except LLMError:
        raise
    except Exception as e:
        log.exception(f"Error during async streaming LLM call: {e}")
        raise LLMError(f"Error during async streaming: {str(e)}")
