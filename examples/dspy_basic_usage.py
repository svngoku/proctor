"""
Basic DSPy Usage Examples

This example demonstrates how to use DSPy-powered techniques in Proctor AI.

DSPy techniques provide:
- Structured outputs with typed fields
- Automatic optimization capabilities
- Better composition and reusability
- Type safety

Run this example:
    python examples/dspy_basic_usage.py
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def example_1_basic_chain_of_thought():
    """Example 1: Basic Chain of Thought with structured output."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Chain of Thought")
    print("=" * 70)

    from proctor.dspy_lm import configure_dspy_with_litellm
    from proctor.dspy_techniques import DSPyChainOfThought

    # Configure DSPy to use LiteLLM
    configure_dspy_with_litellm()

    # Create technique
    cot = DSPyChainOfThought()

    # Solve a problem
    problem = "If a train travels 120 km in 2 hours, what is its average speed?"

    print(f"\nProblem: {problem}")
    print("\n" + "-" * 70)

    # Get structured output
    result = cot.forward(problem=problem)

    print("\n📝 Structured Output:")
    print(f"  Reasoning: {result.reasoning}")
    print(f"  Answer: {result.answer}")

    # You can also use the backward-compatible string interface
    print("\n📄 String Output (backward compatible):")
    string_output = cot.execute(problem)
    print(f"  {string_output}")


def example_2_self_consistency():
    """Example 2: Self-Consistency with multiple reasoning paths."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Self-Consistency (Multiple Reasoning Paths)")
    print("=" * 70)

    from proctor.dspy_lm import configure_dspy_with_litellm
    from proctor.dspy_techniques import DSPySelfConsistency

    configure_dspy_with_litellm()

    # Create self-consistency technique with 5 paths
    sc = DSPySelfConsistency(num_paths=5)

    # Tricky problem where multiple approaches might help
    problem = "A farmer has 17 sheep. All but 9 die. How many sheep are left?"

    print(f"\nProblem: {problem}")
    print("\n" + "-" * 70)

    # Get consensus answer
    result = sc.forward(problem=problem)

    print("\n🎯 Consensus Result:")
    print(f"  Answer: {result.answer}")
    print(f"  Confidence: {result.confidence:.1%}")

    print("\n📊 Vote Distribution:")
    for answer, count in result.vote_distribution.items():
        print(f"  - '{answer}': {count} votes")

    print(f"\n💭 Generated {len(result.reasoning_paths)} reasoning paths")

    # Show first reasoning path as example
    if result.reasoning_paths:
        print("\n  Example reasoning path:")
        print(f"  {result.reasoning_paths[0][:200]}...")


def example_3_comparison():
    """Example 3: Compare classic vs DSPy techniques."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Classic vs DSPy Comparison")
    print("=" * 70)

    from proctor import ChainOfThought as ClassicCoT
    from proctor.dspy_lm import configure_dspy_with_litellm
    from proctor.dspy_techniques import DSPyChainOfThought

    configure_dspy_with_litellm()

    problem = "What are the benefits of renewable energy?"

    print(f"\nProblem: {problem}")
    print("\n" + "-" * 70)

    # Classic version
    print("\n🔵 Classic Chain of Thought:")
    classic_cot = ClassicCoT()
    classic_result = classic_cot.execute(problem)
    print(f"Result: {classic_result[:200]}...")
    print(f"Type: {type(classic_result)}")  # str

    # DSPy version
    print("\n🟢 DSPy Chain of Thought:")
    dspy_cot = DSPyChainOfThought()
    dspy_result = dspy_cot.forward(problem=problem)
    print(f"Reasoning: {dspy_result.reasoning[:200]}...")
    print(f"Answer: {dspy_result.answer[:200]}...")
    print(f"Type: {type(dspy_result)}")  # dspy.Prediction

    print("\n✅ DSPy Benefits:")
    print("  - Structured output (separate fields)")
    print("  - Type-safe access to results")
    print("  - Can be optimized with Teleprompters")
    print("  - Better error handling")
    print("  - Composable with other DSPy modules")


def example_4_typed_access():
    """Example 4: Demonstrate type-safe field access."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Type-Safe Field Access")
    print("=" * 70)

    from proctor.dspy_lm import configure_dspy_with_litellm
    from proctor.dspy_techniques import DSPyChainOfThought

    configure_dspy_with_litellm()

    cot = DSPyChainOfThought()
    problem = "Calculate 15% of 80"

    result = cot.forward(problem=problem)

    print("\n🔍 Accessing Fields:")

    # Type-safe field access
    print(f"\nresult.reasoning:")
    print(f"  {result.reasoning}")

    print(f"\nresult.answer:")
    print(f"  {result.answer}")

    # You can use these in further processing
    print("\n📊 Post-Processing:")
    reasoning_length = len(result.reasoning)
    answer_length = len(result.answer)
    print(f"  Reasoning length: {reasoning_length} characters")
    print(f"  Answer length: {answer_length} characters")

    # Check if answer contains numbers
    has_numbers = any(char.isdigit() for char in result.answer)
    print(f"  Answer contains numbers: {has_numbers}")


def main():
    """Run all examples."""
    # Check if API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n⚠️  OPENROUTER_API_KEY not set!")
        print("Please set your OpenRouter API key in the .env file.")
        print("\nRunning examples without actual LLM calls (demonstrations only)...\n")
        return

    print("\n" + "=" * 70)
    print("PROCTOR AI - DSPy BASIC USAGE EXAMPLES")
    print("=" * 70)

    try:
        example_1_basic_chain_of_thought()
    except Exception as e:
        print(f"\n❌ Example 1 failed: {e}")

    try:
        example_2_self_consistency()
    except Exception as e:
        print(f"\n❌ Example 2 failed: {e}")

    try:
        example_3_comparison()
    except Exception as e:
        print(f"\n❌ Example 3 failed: {e}")

    try:
        example_4_typed_access()
    except Exception as e:
        print(f"\n❌ Example 4 failed: {e}")

    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Try dspy_optimization_example.py for automatic optimization")
    print("  2. Explore creating custom DSPy techniques")
    print("  3. Read DSPY_OPTIMIZATION_PLAN.md for more details")
    print()


if __name__ == "__main__":
    main()
