"""
Utility functions for prompt techniques.
"""
import textwrap
import logging
from typing import Dict, Any, Optional, List
import litellm
from rich.logging import RichHandler
from .config import get_llm_config

# --- Logger Setup ---
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
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

def call_llm(
    prompt: str, 
    system_prompt: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None
) -> str:
    """
    Call the LLM with the given prompt using litellm with openrouter.
    
    Args:
        prompt (str): The user prompt to send
        system_prompt (Optional[str]): Optional system prompt to use
        config_override (Optional[Dict[str, Any]]): Override default config values
        
    Returns:
        str: The LLM response content
    """
    config = get_llm_config()
    
    # Apply config overrides if provided
    if config_override:
        config.update(config_override)
    
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    log.info("Attempting to call LLM...")
    log.debug(f"LLM Config: {config}")
    log.debug(f"Messages: {messages}")
    
    try:
        response = litellm.completion(
            model=config["model"],
            messages=messages,
            api_base=config["api_base"],
            api_key=config["api_key"],
            max_tokens=config.get("max_tokens", 1000),
            temperature=config.get("temperature", 0.7),
        )
        log.debug(f"Raw LLM Response object: {response}")
        # Adjusted to access the content correctly based on common litellm response structure
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content
            log.info("LLM call successful.")
            return content
        else:
            log.error("Received unexpected response format from LLM.")
            log.error(f"Response object: {response}")
            return "Error: Received unexpected response format from LLM."
    except Exception as e:
        log.exception(f"Error calling LLM: {e}")
        return f"Error: {str(e)}" 