"""
Example demonstrating error handling in the proctor package.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from proctor import ZeroShotCoT, get_technique


def demonstrate_error_handling():
    """Demonstrate different error scenarios and how they're handled."""
    print(f"\n{'=' * 50}")
    print("Demonstrating Error Handling")
    print(f"{'=' * 50}")

    # 1. Demonstrate input validation error
    print("\n1. Input Validation Error")
    print("-" * 30)

    technique = ZeroShotCoT()

    try:
        # Passing empty input should raise ValueError
        prompt = technique.generate_prompt("")
        print("Generated prompt:", prompt)
    except ValueError as e:
        print(f"✓ Caught expected ValueError: {e}")

    # 2. Demonstrate API error with invalid API key
    print("\n2. API Error with Invalid Key")
    print("-" * 30)

    # Set an invalid API key to demonstrate API error
    original_api_key = os.environ.get("OPENROUTER_API_KEY", "")

    try:
        # Generate a prompt, which should work even with an invalid key
        prompt = technique.generate_prompt("What is 2+2?")
        print("✓ Generated prompt correctly:", prompt[:30] + "...")

        # Print the actual result from executing with the valid key
        print("\nActual result with valid API key:")
        response = technique.execute("What is 2+2?")
        print("Response:", response[:50] + "..." if len(response) > 50 else response)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    # 3. Demonstrate unknown technique error
    print("\n3. Unknown Technique Error")
    print("-" * 30)

    try:
        # Should return None for non-existent technique
        unknown = get_technique("nonexistent_technique")
        if unknown is None:
            print("✓ get_technique() correctly returned None for unknown technique")
        else:
            print("❌ get_technique() should have returned None")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    print("\nError handling demonstration complete.")


if __name__ == "__main__":
    demonstrate_error_handling()
