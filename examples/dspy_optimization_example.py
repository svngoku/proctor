"""
DSPy Optimization Example

This example demonstrates how to use DSPy's BootstrapFewShot teleprompter
to automatically optimize prompt techniques.

The optimization process:
1. Define a metric to measure quality
2. Provide training examples
3. Let DSPy find the best prompts and examples automatically

Run this example:
    python examples/dspy_optimization_example.py
"""

import os
from dotenv import load_dotenv
import dspy

# Load environment variables
load_dotenv()


def create_training_data():
    """Create training examples for optimization."""
    # Training examples with input/output pairs
    examples = [
        {
            "problem": "What is 25% of 200?",
            "answer": "50"
        },
        {
            "problem": "If a car travels 60 km in 1.5 hours, what is its speed?",
            "answer": "40 km/h"
        },
        {
            "problem": "A store sells apples for $2 each. How much for 7 apples?",
            "answer": "$14"
        },
        {
            "problem": "What is the area of a rectangle with length 8 cm and width 5 cm?",
            "answer": "40 square cm"
        },
        {
            "problem": "If you save $10 per week, how much in 12 weeks?",
            "answer": "$120"
        },
        {
            "problem": "A pizza is cut into 8 slices. You eat 3. How many remain?",
            "answer": "5 slices"
        },
        {
            "problem": "What is 30% of 150?",
            "answer": "45"
        },
        {
            "problem": "A book has 240 pages. You read 60 pages. How many left?",
            "answer": "180 pages"
        },
    ]

    # Convert to DSPy Examples
    trainset = []
    for ex in examples:
        trainset.append(
            dspy.Example(
                problem=ex["problem"],
                answer=ex["answer"]
            ).with_inputs("problem")
        )

    return trainset


def accuracy_metric(example, pred, trace=None):
    """
    Metric to evaluate if the prediction is correct.

    Args:
        example: The ground truth example
        pred: The model's prediction
        trace: Optional execution trace (for debugging)

    Returns:
        float: 1.0 if correct, 0.0 if wrong
    """
    # Simple substring matching (can be made more sophisticated)
    prediction_text = pred.answer.lower().strip()
    expected_text = example.answer.lower().strip()

    # Check if expected answer is in the prediction
    # (allows for slight variations in formatting)
    if expected_text in prediction_text or prediction_text in expected_text:
        return 1.0

    # Also check for numeric equality
    try:
        # Extract numbers
        import re
        pred_numbers = re.findall(r'\d+\.?\d*', prediction_text)
        expected_numbers = re.findall(r'\d+\.?\d*', expected_text)

        if pred_numbers and expected_numbers:
            if float(pred_numbers[0]) == float(expected_numbers[0]):
                return 1.0
    except (ValueError, IndexError):
        pass

    return 0.0


def example_1_basic_optimization():
    """Example 1: Optimize Chain of Thought with BootstrapFewShot."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Optimization with BootstrapFewShot")
    print("=" * 70)

    from proctor.dspy_lm import configure_dspy_with_litellm
    from proctor.dspy_techniques import DSPyChainOfThought

    # Configure DSPy
    configure_dspy_with_litellm()

    # Create training data
    trainset = create_training_data()
    print(f"\n📚 Created {len(trainset)} training examples")

    # Create the technique to optimize
    print("\n🔧 Creating Chain of Thought technique...")
    cot = DSPyChainOfThought()

    # Test before optimization
    print("\n📊 Testing BEFORE optimization:")
    test_problem = "What is 20% of 250?"
    result_before = cot.forward(problem=test_problem)
    print(f"  Problem: {test_problem}")
    print(f"  Answer: {result_before.answer}")

    # Optimize with BootstrapFewShot
    print("\n🚀 Optimizing with BootstrapFewShot...")
    print("  (This will take a moment as it calls the LLM multiple times)")

    teleprompter = dspy.BootstrapFewShot(
        metric=accuracy_metric,
        max_bootstrapped_demos=3,  # Number of examples to bootstrap
        max_labeled_demos=2,        # Number of labeled examples to use
    )

    # Compile the optimized program
    try:
        optimized_cot = teleprompter.compile(
            student=cot,
            trainset=trainset[:5]  # Use subset for faster optimization
        )

        print("\n✅ Optimization complete!")

        # Test after optimization
        print("\n📊 Testing AFTER optimization:")
        result_after = optimized_cot.forward(problem=test_problem)
        print(f"  Problem: {test_problem}")
        print(f"  Answer: {result_after.answer}")

        # Compare
        print("\n📈 Comparison:")
        print(f"  Before: {result_before.answer}")
        print(f"  After:  {result_after.answer}")
        print(f"  Expected: 50")

    except Exception as e:
        print(f"\n⚠️  Optimization failed: {e}")
        print("  This might happen with API rate limits or model issues.")


def example_2_evaluate_performance():
    """Example 2: Evaluate optimized vs non-optimized performance."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Performance Evaluation")
    print("=" * 70)

    from proctor.dspy_lm import configure_dspy_with_litellm
    from proctor.dspy_techniques import DSPyChainOfThought

    configure_dspy_with_litellm()

    # Create train and test sets
    all_data = create_training_data()
    trainset = all_data[:6]
    testset = all_data[6:]

    print(f"\n📚 Dataset split:")
    print(f"  Training: {len(trainset)} examples")
    print(f"  Testing:  {len(testset)} examples")

    # Create and optimize technique
    print("\n🔧 Creating and optimizing technique...")
    cot = DSPyChainOfThought()

    try:
        teleprompter = dspy.BootstrapFewShot(
            metric=accuracy_metric,
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
        )

        optimized_cot = teleprompter.compile(
            student=cot,
            trainset=trainset
        )

        # Evaluate both versions
        print("\n📊 Evaluating on test set...")

        # Non-optimized
        print("\n  Non-optimized CoT:")
        correct_before = 0
        for example in testset:
            pred = cot.forward(problem=example.problem)
            score = accuracy_metric(example, pred)
            correct_before += score
            status = "✓" if score == 1.0 else "✗"
            print(f"    {status} {example.problem[:40]}...")

        accuracy_before = correct_before / len(testset)

        # Optimized
        print("\n  Optimized CoT:")
        correct_after = 0
        for example in testset:
            pred = optimized_cot.forward(problem=example.problem)
            score = accuracy_metric(example, pred)
            correct_after += score
            status = "✓" if score == 1.0 else "✗"
            print(f"    {status} {example.problem[:40]}...")

        accuracy_after = correct_after / len(testset)

        # Results
        print("\n📈 Results:")
        print(f"  Non-optimized accuracy: {accuracy_before:.1%}")
        print(f"  Optimized accuracy:     {accuracy_after:.1%}")
        improvement = accuracy_after - accuracy_before
        print(f"  Improvement:            {improvement:+.1%}")

    except Exception as e:
        print(f"\n⚠️  Evaluation failed: {e}")


def example_3_save_and_load_optimized():
    """Example 3: Save and load optimized techniques."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Save and Load Optimized Techniques")
    print("=" * 70)

    from proctor.dspy_lm import configure_dspy_with_litellm
    from proctor.dspy_techniques import DSPyChainOfThought

    configure_dspy_with_litellm()

    trainset = create_training_data()[:4]

    print("\n🔧 Creating and optimizing technique...")
    cot = DSPyChainOfThought()

    try:
        teleprompter = dspy.BootstrapFewShot(
            metric=accuracy_metric,
            max_bootstrapped_demos=2,
        )

        optimized_cot = teleprompter.compile(
            student=cot,
            trainset=trainset
        )

        # Save to file
        save_path = "/tmp/optimized_cot.json"
        print(f"\n💾 Saving optimized technique to {save_path}...")
        optimized_cot.save(save_path)
        print("  ✅ Saved!")

        # Load from file
        print(f"\n📂 Loading optimized technique from {save_path}...")
        new_cot = DSPyChainOfThought()
        new_cot.load(save_path)
        print("  ✅ Loaded!")

        # Test loaded version
        print("\n🧪 Testing loaded technique:")
        test_problem = "What is 10% of 500?"
        result = new_cot.forward(problem=test_problem)
        print(f"  Problem: {test_problem}")
        print(f"  Answer: {result.answer}")

    except Exception as e:
        print(f"\n⚠️  Save/load failed: {e}")


def main():
    """Run all optimization examples."""
    # Check if API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n⚠️  OPENROUTER_API_KEY not set!")
        print("Please set your OpenRouter API key in the .env file.")
        print("\nThese examples require API access to demonstrate optimization.\n")
        return

    print("\n" + "=" * 70)
    print("PROCTOR AI - DSPy OPTIMIZATION EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate automatic prompt optimization using DSPy.")
    print("Note: These examples make multiple API calls and may take some time.\n")

    try:
        example_1_basic_optimization()
    except Exception as e:
        print(f"\n❌ Example 1 failed: {e}")

    try:
        example_2_evaluate_performance()
    except Exception as e:
        print(f"\n❌ Example 2 failed: {e}")

    try:
        example_3_save_and_load_optimized()
    except Exception as e:
        print(f"\n❌ Example 3 failed: {e}")

    print("\n" + "=" * 70)
    print("OPTIMIZATION EXAMPLES COMPLETE")
    print("=" * 70)
    print("\n🎉 Key Takeaways:")
    print("  - DSPy can automatically optimize prompts")
    print("  - BootstrapFewShot learns from examples")
    print("  - Optimized techniques can be saved and reused")
    print("  - Measure before/after on a held-out set before claiming an improvement")
    print("\nRead DSPY_OPTIMIZATION_PLAN.md for more details!")
    print()


if __name__ == "__main__":
    main()
