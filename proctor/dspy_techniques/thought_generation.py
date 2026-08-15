"""
DSPy-powered Thought Generation techniques.

This module provides DSPy versions of thought generation techniques
like Chain of Thought, Zero-Shot CoT, and Self-Consistency, with
structured outputs and automatic optimization capabilities.
"""

import dspy
from collections import Counter
from ..dspy_base import DSPyPromptTechnique
from ..dspy_signatures import ChainOfThoughtSignature
from ..utils import log


class DSPyChainOfThought(DSPyPromptTechnique):
    """
    DSPy-powered Chain of Thought with structured reasoning.

    This version uses DSPy's ChainOfThought predictor to generate
    structured reasoning with separate fields for the reasoning process
    and final answer.

    Benefits over classic ChainOfThought:
    - Structured output (reasoning + answer fields)
    - Can be optimized with BootstrapFewShot
    - Type-safe composition with other modules
    - Reliable parsing (no string manipulation)

    Example:
        >>> from proctor.dspy_lm import configure_dspy_with_litellm
        >>> configure_dspy_with_litellm()
        >>>
        >>> cot = DSPyChainOfThought()
        >>> result = cot.forward(problem="What is 15% of 80?")
        >>> print(result.reasoning)
        >>> print(result.answer)
    """

    def __init__(self):
        """Initialize DSPy Chain of Thought."""
        super().__init__(
            name="DSPy Chain of Thought",
            identifier="dspy-cot",
            description="Structured CoT with step-by-step reasoning and typed outputs",
        )
        # Use DSPy's built-in ChainOfThought predictor
        self.cot = dspy.ChainOfThought(ChainOfThoughtSignature)

    def forward(self, problem: str, **kwargs) -> dspy.Prediction:
        """
        Execute Chain of Thought reasoning.

        Args:
            problem (str): The problem to solve
            **kwargs: Additional arguments (currently unused)

        Returns:
            dspy.Prediction: Contains 'reasoning' and 'answer' fields

        Example:
            >>> result = cot.forward(problem="If a train travels 120 km in 2 hours, what is its speed?")
            >>> print(result.reasoning)
            "To find speed, divide distance by time. 120 km / 2 hours = 60 km/h"
            >>> print(result.answer)
            "60 km/h"
        """
        return self.cot(problem=problem)

    def execute(self, input_text: str, **kwargs) -> str:
        """
        Backward-compatible string interface.

        Args:
            input_text (str): The problem to solve
            **kwargs: Additional arguments

        Returns:
            str: Formatted string with reasoning and answer
        """
        log.info(f"[DSPy] Executing: [bold magenta]{self.name}[/]")

        result = self.forward(problem=input_text, **kwargs)

        # Format as string for backward compatibility
        output = f"Reasoning: {result.reasoning}\n\nAnswer: {result.answer}"
        log.info(f"[DSPy] Output:\n[green]{output}[/]")

        return output


class DSPyZeroShotCoT(DSPyPromptTechnique):
    """
    DSPy-powered Zero-Shot Chain of Thought.

    Similar to DSPyChainOfThought but emphasizes zero-shot reasoning
    (no examples provided). Uses the same underlying structure but
    can be optimized separately.

    Example:
        >>> zs_cot = DSPyZeroShotCoT()
        >>> result = zs_cot.forward(problem="What is the capital of France?")
        >>> print(result.answer)
    """

    def __init__(self):
        """Initialize DSPy Zero-Shot CoT."""
        super().__init__(
            name="DSPy Zero-Shot CoT",
            identifier="dspy-zero-shot-cot",
            description="Zero-shot reasoning with structured outputs",
        )
        self.predictor = dspy.ChainOfThought(ChainOfThoughtSignature)

    def forward(self, problem: str, **kwargs) -> dspy.Prediction:
        """
        Execute zero-shot chain of thought reasoning.

        Args:
            problem (str): The problem to solve
            **kwargs: Additional arguments

        Returns:
            dspy.Prediction: Contains 'reasoning' and 'answer' fields
        """
        return self.predictor(problem=problem)


class DSPySelfConsistency(DSPyPromptTechnique):
    """
    DSPy-powered Self-Consistency with multiple reasoning paths.

    This technique generates multiple reasoning paths and selects
    the most consistent answer. This is a REAL implementation of
    self-consistency, unlike the classic version which only prompts for it.

    The technique:
    1. Generates N independent reasoning paths
    2. Extracts the answer from each path
    3. Uses majority voting to select the most consistent answer
    4. Returns the consensus with confidence score

    Args:
        num_paths (int): Number of reasoning paths to generate (default: 5)

    Example:
        >>> sc = DSPySelfConsistency(num_paths=5)
        >>> result = sc.forward(problem="A farmer has 17 sheep. All but 9 die. How many are left?")
        >>> print(result.answer)  # "9"
        >>> print(result.confidence)  # 1.0 (if all paths agreed)
        >>> print(len(result.reasoning_paths))  # 5
    """

    def __init__(self, num_paths: int = 5):
        """
        Initialize DSPy Self-Consistency.

        Args:
            num_paths (int): Number of independent reasoning paths
        """
        super().__init__(
            name="DSPy Self-Consistency",
            identifier="dspy-self-consistency",
            description=f"Ensemble reasoning with {num_paths} paths and majority voting",
        )
        self.num_paths = num_paths

        # Create multiple independent CoT predictors
        self.predictors = [
            dspy.ChainOfThought(ChainOfThoughtSignature) for _ in range(num_paths)
        ]

    def forward(self, problem: str, **kwargs) -> dspy.Prediction:
        """
        Execute self-consistency reasoning.

        Args:
            problem (str): The problem to solve
            **kwargs: Additional arguments

        Returns:
            dspy.Prediction: Contains:
                - answer: The consensus answer
                - confidence: Agreement ratio (0.0 to 1.0)
                - reasoning_paths: List of all reasoning processes
                - all_answers: List of all candidate answers
        """
        log.info(
            f"[DSPy] Generating {self.num_paths} reasoning paths for self-consistency"
        )

        # Generate multiple reasoning paths
        results = []
        for i, predictor in enumerate(self.predictors):
            try:
                result = predictor(problem=problem)
                results.append(result)
                log.debug(f"[DSPy] Path {i + 1}: {result.answer}")
            except Exception as e:
                log.warning(f"[DSPy] Path {i + 1} failed: {e}")
                continue

        if not results:
            raise RuntimeError("All reasoning paths failed")

        # Extract answers
        answers = [r.answer for r in results]
        reasoning_paths = [r.reasoning for r in results]

        # Perform majority voting
        answer_counts = Counter(answers)
        most_common_answer, count = answer_counts.most_common(1)[0]

        # Calculate confidence
        confidence = count / len(answers)

        log.info(
            f"[DSPy] Consensus: '{most_common_answer}' (confidence: {confidence:.2f})"
        )

        # Return structured result
        return dspy.Prediction(
            answer=most_common_answer,
            confidence=confidence,
            reasoning_paths=reasoning_paths,
            all_answers=answers,
            vote_distribution=dict(answer_counts),
        )

    def execute(self, input_text: str, **kwargs) -> str:
        """
        Backward-compatible string interface.

        Args:
            input_text (str): The problem to solve
            **kwargs: Additional arguments

        Returns:
            str: Formatted output with consensus answer and metadata
        """
        log.info(f"[DSPy] Executing: [bold magenta]{self.name}[/]")

        result = self.forward(problem=input_text, **kwargs)

        # Format for backward compatibility
        output_lines = [
            f"Consensus Answer: {result.answer}",
            f"Confidence: {result.confidence:.1%}",
            "\nVote Distribution:",
        ]

        for answer, count in result.vote_distribution.items():
            output_lines.append(f"  - '{answer}': {count} votes")

        output_lines.append(f"\nReasoning Paths ({len(result.reasoning_paths)}):")
        for i, reasoning in enumerate(result.reasoning_paths, 1):
            output_lines.append(f"\nPath {i}:")
            output_lines.append(
                f"  {reasoning[:200]}..." if len(reasoning) > 200 else f"  {reasoning}"
            )

        output = "\n".join(output_lines)
        log.info(f"[DSPy] Output:\n[green]{output}[/]")

        return output
