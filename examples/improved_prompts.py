"""
Example demonstrating the improved prompts in the proctor package.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from proctor import RolePrompting, ZeroShotCoT, DECOMP, SelfConsistency

# Check if API key is set
API_KEY_SET = bool(os.environ.get("OPENROUTER_API_KEY"))


def demonstrate_improved_role_prompting():
    """Demonstrate the improved RolePrompting technique."""
    print(f"\n{'=' * 50}")
    print("Improved Role Prompting Example")
    print(f"{'=' * 50}")

    role_prompt = RolePrompting()

    # Define a problem that would benefit from expert perspective
    problem = "Explain the potential implications of quantum computing for cybersecurity in the next decade."

    # Generate basic prompt (default parameters)
    basic_prompt = role_prompt.generate_prompt(problem)

    print("\nBasic Role Prompt (Default Parameters):")
    print(f"{'~' * 50}")
    print(basic_prompt)
    print(f"{'~' * 50}")

    # Generate advanced prompt with custom parameters
    advanced_prompt = role_prompt.generate_prompt(
        problem,
        role="cryptographer",
        field="quantum computing and cybersecurity",
        experience="senior",
        audience="corporate security executives",
    )

    print("\nAdvanced Role Prompt (Custom Parameters):")
    print(f"{'~' * 50}")
    print(advanced_prompt)
    print(f"{'~' * 50}")

    # Execute if API key is set
    if API_KEY_SET:
        print("\nCalling LLM with advanced prompt...")
        response = role_prompt.execute(
            problem,
            role="cryptographer",
            field="quantum computing and cybersecurity",
            experience="senior",
            audience="corporate security executives",
        )
        print("\nLLM Response:")
        print(f"{'~' * 50}")
        print(response)
        print(f"{'~' * 50}")
    else:
        print("\nSkipping LLM execution: OPENROUTER_API_KEY not set.")


def demonstrate_improved_zero_shot_cot():
    """Demonstrate the improved ZeroShotCoT technique."""
    print(f"\n{'=' * 50}")
    print("Improved Zero-Shot Chain-of-Thought Example")
    print(f"{'=' * 50}")

    cot_prompt = ZeroShotCoT()

    # Define a problem that requires reasoning
    problem = "A train travels west at 60 mph from New York and another train travels east at 40 mph from Chicago. If the cities are 800 miles apart and both trains depart at the same time, how long will it take for them to meet?"

    # Generate basic prompt
    basic_prompt = cot_prompt.generate_prompt(problem)

    print("\nBasic Zero-Shot CoT Prompt:")
    print(f"{'~' * 50}")
    print(basic_prompt)
    print(f"{'~' * 50}")

    # Generate advanced prompt with custom parameters
    advanced_prompt = cot_prompt.generate_prompt(
        problem,
        domain="physics and kinematics",
        reasoning_style="mathematical",
        complexity="intermediate",
    )

    print("\nAdvanced Zero-Shot CoT Prompt:")
    print(f"{'~' * 50}")
    print(advanced_prompt)
    print(f"{'~' * 50}")

    # Execute if API key is set
    if API_KEY_SET:
        print("\nCalling LLM with advanced prompt...")
        response = cot_prompt.execute(
            problem,
            domain="physics and kinematics",
            reasoning_style="mathematical",
            complexity="intermediate",
        )
        print("\nLLM Response:")
        print(f"{'~' * 50}")
        print(response)
        print(f"{'~' * 50}")
    else:
        print("\nSkipping LLM execution: OPENROUTER_API_KEY not set.")


def demonstrate_improved_decomposition():
    """Demonstrate the improved DECOMP technique."""
    print(f"\n{'=' * 50}")
    print("Improved Decomposition Technique Example")
    print(f"{'=' * 50}")

    decomp_prompt = DECOMP()

    # Define a complex problem
    problem = "Design a strategy for a mid-sized company to reduce its carbon footprint by 30% over the next 5 years."

    # Generate advanced prompt with custom parameters
    advanced_prompt = decomp_prompt.generate_prompt(
        problem,
        num_subproblems=4,
        approach="hierarchical",
        domain="sustainable business management",
        clear_dependencies=True,
    )

    print("\nAdvanced Decomposition Prompt:")
    print(f"{'~' * 50}")
    print(advanced_prompt)
    print(f"{'~' * 50}")

    # Execute if API key is set
    if (
        API_KEY_SET and False
    ):  # Set to True to actually execute (disabled to save tokens)
        print("\nCalling LLM with advanced prompt...")
        response = decomp_prompt.execute(
            problem,
            num_subproblems=4,
            approach="hierarchical",
            domain="sustainable business management",
            clear_dependencies=True,
        )
        print("\nLLM Response:")
        print(f"{'~' * 50}")
        print(response)
        print(f"{'~' * 50}")
    else:
        print("\nSkipping LLM execution for this example to save tokens.")


def demonstrate_improved_self_consistency():
    """Demonstrate the improved SelfConsistency technique."""
    print(f"\n{'=' * 50}")
    print("Improved Self-Consistency Technique Example")
    print(f"{'=' * 50}")

    sc_prompt = SelfConsistency()

    # Define a problem that might benefit from multiple approaches
    problem = "What are the most effective strategies for addressing climate change at the individual, corporate, and governmental levels?"

    # Generate advanced prompt with custom parameters
    advanced_prompt = sc_prompt.generate_prompt(
        problem,
        num_paths=3,
        approach_diversity=True,
        domain="environmental policy",
        include_metacognition=True,
        reasoning_styles=["analytical", "systems thinking", "economic"],
        path_length="standard",
    )

    print("\nAdvanced Self-Consistency Prompt:")
    print(f"{'~' * 50}")
    print(advanced_prompt)
    print(f"{'~' * 50}")

    # Execute if API key is set
    if (
        API_KEY_SET and False
    ):  # Set to True to actually execute (disabled to save tokens)
        print("\nCalling LLM with advanced prompt...")
        response = sc_prompt.execute(
            problem,
            num_paths=3,
            approach_diversity=True,
            domain="environmental policy",
            include_metacognition=True,
            reasoning_styles=["analytical", "systems thinking", "economic"],
            path_length="standard",
        )
        print("\nLLM Response:")
        print(f"{'~' * 50}")
        print(response)
        print(f"{'~' * 50}")
    else:
        print("\nSkipping LLM execution for this example to save tokens.")


if __name__ == "__main__":
    demonstrate_improved_role_prompting()
    demonstrate_improved_zero_shot_cot()
    demonstrate_improved_decomposition()
    demonstrate_improved_self_consistency()
