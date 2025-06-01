"""
Basic usage examples for the proctor package.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file in the current directory (proctor/)
load_dotenv()

# Assuming the package is installed or PYTHONPATH is set correctly
# If running directly from the repo root, you might need path adjustments
# Or install the package using `uv pip install .` in the `proctor/` directory.
from proctor import (
    get_technique,  # Added get_technique for demonstration
)

# --- IMPORTANT ---
# Replace with your actual OpenRouter API key or set the environment variable
# os.environ["OPENROUTER_API_KEY"] = "YOUR_OPENROUTER_API_KEY"
# It's recommended to use the .env file (created in the same directory)
# instead of setting the variable directly in the code.
# --- IMPORTANT ---

# Check if the API key is set, otherwise skip execution
API_KEY_SET = bool(os.environ.get("OPENROUTER_API_KEY"))


def main():
    # Example problem
    problem = "A train travels from city A to city B at 60 km/h. Another train travels from city B to city A at 90 km/h. The cities are 300 km apart. If both trains depart at the same time, how far from city A will they meet?"

    # Techniques to demonstrate
    technique_names = [
        "emotion_prompting",
        "role_prompting",
        "chain_of_thought",
        "zero_shot_cot",
        "decomp",
    ]

    for name in technique_names:
        technique = get_technique(name)
        if not technique:
            print(f"\nSkipping unknown technique: {name}")
            continue

        print(f"\n{'=' * 50}")
        print(f"Using technique: {technique.name} ({technique.identifier})")
        print(f"{'=' * 50}")

        # Generate the prompt
        # Pass specific args if needed, e.g., emotion="calm" for EmotionPrompting
        prompt = technique.generate_prompt(problem)
        print("\nGenerated Prompt:")
        print(f"{'~' * 50}")
        print(prompt)
        print(f"{'~' * 50}")

        # Execute the technique to get the LLM response (if API key is set)
        if API_KEY_SET:
            print("\nCalling LLM (requires OPENROUTER_API_KEY to be set)...")
            # You can override LLM config per call if needed:
            # llm_config_override = {"model": "anthropic/claude-3-haiku"}
            # response = technique.execute(problem, llm_config=llm_config_override)
            response = technique.execute(problem)
            print("\nLLM Response:")
            print(f"{'~' * 50}")
            print(response)
            print(f"{'~' * 50}")
        else:
            print(
                "\nSkipping LLM execution: OPENROUTER_API_KEY environment variable not set."
            )
            print("(Set the variable and re-run to see LLM responses)")

    print("\nDone!")


if __name__ == "__main__":
    # Load environment variables before checking API_KEY_SET
    load_dotenv()
    API_KEY_SET = bool(
        os.environ.get("OPENROUTER_API_KEY")
    )  # Re-check after loading .env

    # Add a check for the API key before running main
    if (
        not API_KEY_SET or os.environ.get("OPENROUTER_API_KEY") == "YOUR_API_KEY_HERE"
    ):  # Also check placeholder
        print(
            "Warning: OPENROUTER_API_KEY environment variable is not set or is placeholder."
        )
        print("LLM execution will be skipped in the examples.")
        print(
            "Please set it in the '.env' file if you want to interact with the OpenRouter API."
        )
    main()
