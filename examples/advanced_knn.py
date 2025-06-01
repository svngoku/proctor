"""
Example of using the advanced KNN implementation.

IMPORTANT: This requires additional dependencies.
Install with: uv pip install -e ".[knn]"
"""

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to import the advanced KNN implementation
try:
    from proctor.few_shot.knn_implementation import SemanticKNN, _DEPS_AVAILABLE
    from proctor import KNN
except ImportError as e:
    print(f"Import error: {e}")
    print("This example requires additional dependencies.")
    print('Install with: uv pip install -e ".[knn]"')
    exit(1)


def main():
    """Run the advanced KNN example."""
    print(f"\n{'=' * 50}")
    print("Advanced KNN Implementation Example")
    print(f"{'=' * 50}")

    # Check if the required dependencies are available
    if not _DEPS_AVAILABLE:
        print("Required dependencies not available.")
        print("Install with: uv pip install sentence-transformers scikit-learn")
        return

    # Create a pool of examples with diverse topics
    examples_pool = [
        {
            "input": "What is machine learning?",
            "output": "Machine learning is a branch of artificial intelligence that focuses on building systems that can learn from and make decisions based on data.",
        },
        {
            "input": "Explain quantum computing",
            "output": "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, allowing certain computations to be performed exponentially faster than classical computers.",
        },
        {
            "input": "How does photosynthesis work?",
            "output": "Photosynthesis is the process by which plants convert light energy into chemical energy. They use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
        },
        {
            "input": "What is the theory of relativity?",
            "output": "Einstein's theory of relativity consists of special relativity (which relates space and time) and general relativity (which describes gravity as a geometric property of spacetime).",
        },
        {
            "input": "How do vaccines work?",
            "output": "Vaccines work by training the immune system to recognize and combat pathogens. They contain weakened or inactive parts of a specific organism that triggers an immune response.",
        },
        {
            "input": "What are black holes?",
            "output": "Black holes are regions of spacetime where gravity is so strong that nothing—no particles or even electromagnetic radiation such as light—can escape from it.",
        },
        {
            "input": "Explain how blockchain works",
            "output": "Blockchain is a distributed ledger technology that records transactions across many computers. Each block contains a timestamp and link to the previous block, forming a chain that is resistant to modification.",
        },
    ]

    # Sample query about artificial intelligence
    query = "How does deep learning work?"

    print(f"\nQuery: {query}")
    print(f"\n{'-' * 50}")
    print("Using standard KNN technique (random sampling):")

    # Use the standard KNN technique (which currently uses random sampling)
    standard_knn = KNN()
    standard_prompt = standard_knn.generate_prompt(
        query, examples_pool=examples_pool, k=3
    )

    print("\nGenerated Prompt:")
    print(f"{'~' * 50}")
    print(standard_prompt)
    print(f"{'~' * 50}")

    print(f"\n{'-' * 50}")
    print("Using advanced Semantic KNN (with proper similarity calculation):")

    try:
        # Initialize the advanced KNN
        semantic_knn = SemanticKNN()

        # Find the most similar examples to the query
        similar_examples = semantic_knn.find_nearest(query, examples_pool)

        print("\nNearest neighbors with similarity scores:")
        for example, similarity in similar_examples:
            print(f"- Similarity: {similarity:.4f}")
            print(f"  Input: {example['input']}")
            print(f"  Output: {example['output'][:50]}...\n")

        # Generate a prompt using these examples
        examples_text = "\n\n".join(
            [
                f"Input: {example['input']}\nOutput: {example['output']}"
                for example, _ in similar_examples
            ]
        )

        advanced_prompt = f"""
Here are some examples that are semantically similar to your query:

{examples_text}

Now, for your query:
Input: {query}
Output:
        """

        print("\nGenerated Prompt using Semantic KNN:")
        print(f"{'~' * 50}")
        print(advanced_prompt)
        print(f"{'~' * 50}")

    except Exception as e:
        print(f"Error running advanced KNN: {e}")


if __name__ == "__main__":
    main()
