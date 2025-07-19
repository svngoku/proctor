# OpenRouter Model Reference Guide

## Important: Model Naming Convention

When using OpenRouter with LiteLLM, **DO NOT** prefix model names with `openrouter/`. LiteLLM automatically detects OpenRouter from the `api_base` URL.

## Available Models on OpenRouter

### Google Models
- `google/gemini-2.0-flash-exp` - Latest Gemini experimental
- `google/gemini-2.0-flash-thinking-exp` - Gemini with thinking mode
- `google/gemini-2.5-flash` - Gemini 2.5 Flash
- `google/gemini-pro` - Gemini Pro
- `google/gemini-pro-vision` - Gemini Pro with vision

### Anthropic Models
- `anthropic/claude-3.5-sonnet` - Latest Claude 3.5 Sonnet
- `anthropic/claude-3.5-haiku` - Latest Claude 3.5 Haiku
- `anthropic/claude-3-opus` - Claude 3 Opus (most capable)
- `anthropic/claude-3-sonnet` - Claude 3 Sonnet
- `anthropic/claude-3-haiku` - Claude 3 Haiku

### DeepSeek Models
- `deepseek/deepseek-r1` - DeepSeek R1 (reasoning model)
- `deepseek/deepseek-chat` - DeepSeek Chat

### Meta (Llama) Models
- `meta-llama/llama-3.3-70b-instruct` - Llama 3.3 70B
- `meta-llama/llama-3.2-90b-vision-instruct` - Llama 3.2 90B with vision
- `meta-llama/llama-3.2-11b-vision-instruct` - Llama 3.2 11B with vision
- `meta-llama/llama-3.2-3b-instruct` - Llama 3.2 3B
- `meta-llama/llama-3.2-1b-instruct` - Llama 3.2 1B
- `meta-llama/llama-3.1-405b-instruct` - Llama 3.1 405B
- `meta-llama/llama-3.1-70b-instruct` - Llama 3.1 70B
- `meta-llama/llama-3.1-8b-instruct` - Llama 3.1 8B

### Mistral Models
- `mistralai/mistral-large` - Mistral Large
- `mistralai/mistral-medium` - Mistral Medium
- `mistralai/mistral-small` - Mistral Small
- `mistralai/mixtral-8x7b-instruct` - Mixtral 8x7B
- `mistralai/mixtral-8x22b-instruct` - Mixtral 8x22B

### OpenAI Models
- `openai/gpt-4o` - GPT-4o
- `openai/gpt-4o-mini` - GPT-4o Mini
- `openai/gpt-4-turbo` - GPT-4 Turbo
- `openai/gpt-4` - GPT-4
- `openai/gpt-3.5-turbo` - GPT-3.5 Turbo

### Other Notable Models
- `x-ai/grok-2` - Grok 2
- `x-ai/grok-2-mini` - Grok 2 Mini
- `cohere/command-r-plus` - Command R+
- `cohere/command-r` - Command R
- `perplexity/llama-3.1-sonar-large-128k-online` - Perplexity Sonar
- `qwen/qwen-2.5-72b-instruct` - Qwen 2.5 72B

## Example Usage

```python
from dotenv import load_dotenv
import os

load_dotenv()

# Correct configuration
config = {
    "model": "google/gemini-2.5-flash",  # No 'openrouter/' prefix!
    "api_base": "https://openrouter.ai/api/v1",
    "api_key": os.environ.get("OPENROUTER_API_KEY"),
    "temperature": 0.7,
    "max_tokens": 1000,
}

# WRONG - Do not do this:
# "model": "openrouter/google/gemini-2.5-flash"  # This will cause errors!
```

## Debugging Tips

If you see errors like:
- `LiteLLM completion() model= google/gemini-2.5-flash; provider = openrouter`
- Model routing issues

This means LiteLLM is correctly detecting OpenRouter from the `api_base`. The model name should NOT have the `openrouter/` prefix.

## Setting Environment Variables

In your `.env` file:
```bash
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here
```

## Full Working Example

```python
from proctor import ChainOfThought

# Use any model from the list above
technique = ChainOfThought()
response = technique.execute(
    "What is the meaning of life?",
    llm_config={
        "model": "anthropic/claude-3.5-sonnet",  # Correct format
        "api_base": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "temperature": 0.7,
        "max_tokens": 1000,
    }
)
```
