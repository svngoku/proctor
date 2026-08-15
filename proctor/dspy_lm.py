"""
DSPy language-model adapter over Proctor's OpenRouter client.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import dspy

from .config import get_llm_config
from .utils import LLMError, call_llm, call_llm_async, log


def _flatten_messages(messages: List[Dict[str, str]]) -> tuple[Optional[str], str]:
    system_prompt = None
    user_parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content
        elif role == "assistant":
            user_parts.append(f"Assistant: {content}")
        else:
            user_parts.append(content)
    return system_prompt, "\n".join(user_parts)


def _openai_response(text: str, model: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None), logprobs=None
            )
        ],
        usage={},
        model=model,
    )


class LiteLLMLanguageModel(dspy.BaseLM):
    """Route DSPy LM calls through Proctor's OpenRouter helpers."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs,
    ):
        config = get_llm_config()
        model_name = model or config.get("model", "openai/gpt-4o")
        super().__init__(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base or config.get("api_base"),
            api_key=api_key or config.get("api_key"),
            **kwargs,
        )
        self.model_name = model_name
        log.info(f"[DSPy] Initialized language model: {self.model_name}")

    def _merged_config(self, **kwargs) -> Dict[str, Any]:
        return {
            "model": self.model,
            "api_base": self.kwargs.get("api_base"),
            "api_key": self.kwargs.get("api_key"),
            "max_tokens": kwargs.get("max_tokens", self.kwargs.get("max_tokens", 1000)),
            "temperature": kwargs.get(
                "temperature", self.kwargs.get("temperature", 0.7)
            ),
        }

    def _to_prompt(
        self, prompt: Optional[str], messages: Optional[List[Dict[str, str]]]
    ) -> tuple[Optional[str], str]:
        if messages:
            return _flatten_messages(messages)
        return None, prompt or ""

    def forward(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ):
        system_prompt, text = self._to_prompt(prompt, messages)
        try:
            response = call_llm(
                prompt=text,
                system_prompt=system_prompt,
                config_override=self._merged_config(**kwargs),
                max_retries=2,
            )
        except LLMError:
            log.error("[DSPy] LLM call failed")
            raise
        return _openai_response(response, self.model)

    async def aforward(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ):
        system_prompt, text = self._to_prompt(prompt, messages)
        try:
            response = await call_llm_async(
                prompt=text,
                system_prompt=system_prompt,
                config_override=self._merged_config(**kwargs),
                max_retries=2,
            )
        except LLMError:
            log.error("[DSPy] Async LLM call failed")
            raise
        return _openai_response(response, self.model)

    def __deepcopy__(self, memo):
        return self


def configure_dspy_with_litellm(
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> LiteLLMLanguageModel:
    lm = LiteLLMLanguageModel(model=model, api_base=api_base, api_key=api_key, **kwargs)
    dspy.settings.configure(lm=lm)
    log.info("[DSPy] Configured DSPy with OpenRouter backend")
    return lm
