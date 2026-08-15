"""Minimal single-call demo using the OpenRouter SDK backend."""

import os

from dotenv import load_dotenv

from proctor import ChainOfThought, OPENROUTER_BASE_URL
from proctor.models import GPT_5_6_TERRA

MODEL = GPT_5_6_TERRA


def main():
    load_dotenv()

    technique = ChainOfThought()
    problem = "If a shirt costs $40 after a 20% discount, what was the original price?"

    print(f"Technique: {technique.name} ({technique.identifier})")
    print(f"Model: {MODEL}\n")

    response = technique.execute(
        problem,
        llm_config={
            "model": MODEL,
            "api_base": OPENROUTER_BASE_URL,
            "api_key": os.environ.get("OPENROUTER_API_KEY"),
            "temperature": 0.3,
            "max_tokens": 500,
        },
    )
    print("Response:\n" + "~" * 50)
    print(response)


if __name__ == "__main__":
    main()
