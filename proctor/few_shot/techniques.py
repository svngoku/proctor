"""
Implementation of Few-Shot prompting techniques.
"""

from typing import List, Dict, Optional
import random  # Added for KNN example
from ..base import PromptTechnique
from ..utils import dedent_prompt


class ExampleGeneration(PromptTechnique):
    """
    Example Generation prompting generates examples for few-shot learning.
    """

    def __init__(self):
        """Initialize Example Generation technique."""
        super().__init__(
            name="Example Generation",
            identifier="2.2.1.1",
            description="Generates examples for few-shot learning.",
        )

    def generate_prompt(
        self, input_text: str, examples: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> str:
        """
        Generate a few-shot prompt with examples.

        Args:
            input_text (str): Input text
            examples (Optional[List[Dict[str, str]]]): List of example dictionaries with 'input' and 'output' keys.
                                                    Defaults to predefined examples if None.
            **kwargs: Additional arguments

        Returns:
            str: Generated few-shot prompt
        """
        if examples is None:
            examples = [
                {"input": "Example input 1", "output": "Example output 1"},
                {"input": "Example input 2", "output": "Example output 2"},
                {"input": "Example input 3", "output": "Example output 3"},
            ]

        examples_text = "\n\n".join(
            [
                f"Input: {example['input']}\nOutput: {example['output']}"
                for example in examples
            ]
        )

        prompt = dedent_prompt(f"""
        I'll show you some examples of how to solve this type of problem:

        {examples_text}

        Now, please solve the following:
        Input: {input_text}
        Output:
        """)
        return prompt


class ExampleOrdering(PromptTechnique):
    """
    Example Ordering technique that strategically orders examples for optimal learning.

    This technique focuses on the sequence in which examples are presented,
    which can significantly impact few-shot learning performance.
    """

    def __init__(self):
        """Initialize Example Ordering technique."""
        super().__init__(
            name="Example Ordering",
            identifier="2.2.1.1",
            description="Strategically orders examples for optimal few-shot learning.",
        )

    def generate_prompt(
        self,
        input_text: str,
        examples: Optional[List[Dict[str, str]]] = None,
        ordering_strategy: str = "difficulty",
        **kwargs,
    ) -> str:
        """
        Generate a few-shot prompt with strategically ordered examples.

        Args:
            input_text (str): Input text
            examples (Optional[List[Dict[str, str]]]): List of examples to order
            ordering_strategy (str): Strategy for ordering ("difficulty", "similarity", "diversity", "random")
            **kwargs: Additional arguments
                - reverse_order (bool): Whether to reverse the final order

        Returns:
            str: Generated prompt with ordered examples
        """
        if examples is None:
            examples = [
                {"input": "Simple example", "output": "Simple answer", "difficulty": 1},
                {
                    "input": "Moderate example",
                    "output": "Moderate answer",
                    "difficulty": 2,
                },
                {
                    "input": "Complex example",
                    "output": "Complex answer",
                    "difficulty": 3,
                },
            ]

        reverse_order = kwargs.get("reverse_order", False)

        # Order examples based on strategy
        if ordering_strategy == "difficulty":
            # Order by difficulty (easy to hard)
            ordered_examples = sorted(examples, key=lambda x: x.get("difficulty", 1))
        elif ordering_strategy == "similarity":
            # In a full implementation, this would use semantic similarity to input_text
            # For now, using original order as placeholder
            ordered_examples = examples.copy()
        elif ordering_strategy == "diversity":
            # Maximize diversity between consecutive examples
            ordered_examples = examples.copy()
            random.shuffle(ordered_examples)
        elif ordering_strategy == "random":
            ordered_examples = examples.copy()
            random.shuffle(ordered_examples)
        else:
            ordered_examples = examples.copy()

        if reverse_order:
            ordered_examples.reverse()

        examples_text = "\n\n".join(
            [
                f"Example {i + 1}:\nInput: {example['input']}\nOutput: {example['output']}"
                for i, example in enumerate(ordered_examples)
            ]
        )

        prompt = dedent_prompt(f"""
        Here are examples arranged in optimal order for learning:

        {examples_text}

        Now apply the same pattern to:
        Input: {input_text}
        Output:
        """)
        return prompt


class ExemplarSelection(PromptTechnique):
    """
    Exemplar Selection technique that carefully chooses the best examples.

    This technique focuses on selecting the most informative and representative
    examples from a larger pool to maximize few-shot learning effectiveness.
    """

    def __init__(self):
        """Initialize Exemplar Selection technique."""
        super().__init__(
            name="Exemplar Selection",
            identifier="2.2.1.2",
            description="Selects the most informative examples for few-shot learning.",
        )

    def generate_prompt(
        self,
        input_text: str,
        examples_pool: Optional[List[Dict[str, str]]] = None,
        selection_criteria: str = "diversity",
        num_examples: int = 3,
        **kwargs,
    ) -> str:
        """
        Generate a few-shot prompt with carefully selected exemplars.

        Args:
            input_text (str): Input text
            examples_pool (Optional[List[Dict[str, str]]]): Pool of available examples
            selection_criteria (str): Criteria for selection ("diversity", "quality", "relevance", "coverage")
            num_examples (int): Number of examples to select
            **kwargs: Additional arguments
                - quality_threshold (float): Minimum quality score for examples

        Returns:
            str: Generated prompt with selected exemplars
        """
        if examples_pool is None:
            examples_pool = [
                {
                    "input": "High quality example",
                    "output": "Excellent answer",
                    "quality": 0.9,
                },
                {
                    "input": "Medium quality example",
                    "output": "Good answer",
                    "quality": 0.7,
                },
                {
                    "input": "Diverse example",
                    "output": "Different approach",
                    "quality": 0.8,
                },
                {
                    "input": "Relevant example",
                    "output": "Targeted answer",
                    "quality": 0.85,
                },
                {
                    "input": "Coverage example",
                    "output": "Comprehensive answer",
                    "quality": 0.75,
                },
            ]

        quality_threshold = kwargs.get("quality_threshold", 0.0)

        # Filter by quality threshold first
        filtered_examples = [
            ex for ex in examples_pool if ex.get("quality", 1.0) >= quality_threshold
        ]

        # Select based on criteria
        if selection_criteria == "diversity":
            # Select diverse examples (simplified - would use embeddings in full implementation)
            selected = random.sample(
                filtered_examples, min(num_examples, len(filtered_examples))
            )
        elif selection_criteria == "quality":
            # Select highest quality examples
            selected = sorted(
                filtered_examples, key=lambda x: x.get("quality", 0), reverse=True
            )[:num_examples]
        elif selection_criteria == "relevance":
            # Select most relevant to input (simplified)
            selected = filtered_examples[:num_examples]
        elif selection_criteria == "coverage":
            # Select examples that cover different aspects
            selected = filtered_examples[:num_examples]
        else:
            selected = filtered_examples[:num_examples]

        examples_text = "\n\n".join(
            [
                f"Input: {example['input']}\nOutput: {example['output']}"
                for example in selected
            ]
        )

        prompt = dedent_prompt(f"""
        Here are carefully selected examples that best demonstrate the task:

        {examples_text}

        Following the same pattern, solve:
        Input: {input_text}
        Output:
        """)
        return prompt


class SGICL(PromptTechnique):
    """
    Selective Generation In-Context Learning (SG-ICL).

    This technique selectively generates parts of examples during in-context learning,
    focusing on the most informative portions rather than complete examples.
    """

    def __init__(self):
        """Initialize SG-ICL technique."""
        super().__init__(
            name="Selective Generation In-Context Learning (SG-ICL)",
            identifier="2.2.1.2",
            description="Selectively generates informative portions of examples during in-context learning.",
        )

    def generate_prompt(
        self,
        input_text: str,
        examples: Optional[List[Dict[str, str]]] = None,
        focus_parts: List[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate a prompt with selective example generation.

        Args:
            input_text (str): Input text
            examples (Optional[List[Dict[str, str]]]): Base examples
            focus_parts (List[str]): Parts to focus on ("reasoning", "key_steps", "conclusion", "method")
            **kwargs: Additional arguments
                - generation_style (str): Style of selective generation

        Returns:
            str: Generated SG-ICL prompt
        """
        if examples is None:
            examples = [
                {
                    "input": "Solve: 2x + 5 = 13",
                    "output": "x = 4",
                    "reasoning": "Subtract 5, then divide by 2",
                },
                {
                    "input": "Find area of circle r=3",
                    "output": "28.27",
                    "reasoning": "Use π×r² formula",
                },
            ]

        if focus_parts is None:
            focus_parts = ["reasoning", "key_steps"]

        generation_style = kwargs.get("generation_style", "step_by_step")

        # Generate selective examples
        selective_examples = []
        for example in examples:
            selective_example = f"Input: {example['input']}\n"

            if "reasoning" in focus_parts and "reasoning" in example:
                selective_example += f"Key reasoning: {example['reasoning']}\n"
            if "key_steps" in focus_parts:
                selective_example += "[Focus on the critical steps]\n"
            if "method" in focus_parts:
                selective_example += "[Method/approach highlighted]\n"

            selective_example += f"Output: {example['output']}"
            selective_examples.append(selective_example)

        examples_text = "\n\n".join(selective_examples)

        prompt = dedent_prompt(f"""
        I'll show you examples with focus on the most important parts ({", ".join(focus_parts)}):

        {examples_text}

        Now, applying the same focused approach:
        Input: {input_text}
        [Generate with selective focus on: {", ".join(focus_parts)}]
        Output:
        """)
        return prompt


class VoteK(PromptTechnique):
    """
    Vote-K technique that generates multiple responses and selects the best through voting.

    This technique generates k different responses to the same prompt and uses
    various voting mechanisms to select or combine the best response.
    """

    def __init__(self):
        """Initialize Vote-K technique."""
        super().__init__(
            name="Vote-K",
            identifier="2.2.1.2",
            description="Generates multiple responses and selects the best through voting mechanisms.",
        )

    def generate_prompt(
        self, input_text: str, k: int = 3, voting_method: str = "majority", **kwargs
    ) -> str:
        """
        Generate a prompt for Vote-K technique.

        Args:
            input_text (str): Input text
            k (int): Number of responses to generate
            voting_method (str): Voting method ("majority", "quality", "consensus", "ensemble")
            **kwargs: Additional arguments
                - examples (List[Dict]): Optional examples for few-shot learning
                - confidence_weights (bool): Whether to use confidence weighting

        Returns:
            str: Generated Vote-K prompt
        """
        examples = kwargs.get("examples", [])
        confidence_weights = kwargs.get("confidence_weights", False)

        examples_text = ""
        if examples:
            examples_text = (
                "\n\n".join(
                    [
                        f"Input: {example['input']}\nOutput: {example['output']}"
                        for example in examples
                    ]
                )
                + "\n\n"
            )

        confidence_instruction = ""
        if confidence_weights:
            confidence_instruction = (
                "\n- Provide a confidence score (0-1) for each response"
            )

        voting_instructions = {
            "majority": "Select the response that appears most frequently or is most similar to others",
            "quality": "Select the highest quality response based on accuracy and completeness",
            "consensus": "Find the response that best represents the consensus of all attempts",
            "ensemble": "Combine insights from all responses into a comprehensive answer",
        }.get(voting_method, "Select the best response using your judgment")

        prompt = dedent_prompt(f"""
        {examples_text}I want you to approach this problem using a Vote-K method with k={k}.

        Task: {input_text}

        Please:
        1. Generate {k} different responses to this task
        2. For each response, think through the problem independently
        3. After generating all {k} responses, {voting_instructions}{confidence_instruction}

        Format your response as:

        Response 1: [Your first response]
        Response 2: [Your second response]
        Response 3: [Your third response]
        {"..." if k > 3 else ""}

        Final Selection: [The best response based on {voting_method} voting]
        Reasoning: [Why this response was selected]
        """)
        return prompt


class PromptMining(PromptTechnique):
    """
    Prompt Mining technique that discovers and utilizes effective prompt patterns.

    This technique focuses on identifying, extracting, and applying successful
    prompt patterns from examples or previous interactions.
    """

    def __init__(self):
        """Initialize Prompt Mining technique."""
        super().__init__(
            name="Prompt Mining",
            identifier="2.2.1.2",
            description="Discovers and utilizes effective prompt patterns for improved performance.",
        )

    def generate_prompt(
        self,
        input_text: str,
        pattern_examples: Optional[List[Dict[str, str]]] = None,
        mining_focus: str = "structure",
        **kwargs,
    ) -> str:
        """
        Generate a prompt using mined patterns.

        Args:
            input_text (str): Input text
            pattern_examples (Optional[List[Dict[str, str]]]): Examples containing successful patterns
            mining_focus (str): Focus area for pattern mining ("structure", "phrasing", "strategy", "format")
            **kwargs: Additional arguments
                - pattern_templates (List[str]): Explicit pattern templates to use
                - adaptive (bool): Whether to adapt patterns to the current task

        Returns:
            str: Generated prompt using mined patterns
        """
        if pattern_examples is None:
            pattern_examples = [
                {
                    "input": "Analyze this problem step by step",
                    "output": "1. Identify key components 2. Analyze relationships 3. Draw conclusions",
                    "pattern": "step_by_step_analysis",
                },
                {
                    "input": "Think about this from multiple perspectives",
                    "output": "Perspective 1: ... Perspective 2: ... Synthesis: ...",
                    "pattern": "multi_perspective",
                },
            ]

        pattern_templates = kwargs.get("pattern_templates", [])
        adaptive = kwargs.get("adaptive", True)

        # Mine patterns based on focus area
        if mining_focus == "structure":
            mined_pattern = "Follow a clear structured approach with distinct phases"
        elif mining_focus == "phrasing":
            mined_pattern = "Use clear, directive language with specific action words"
        elif mining_focus == "strategy":
            mined_pattern = (
                "Apply proven problem-solving strategies from successful examples"
            )
        elif mining_focus == "format":
            mined_pattern = (
                "Structure the response in a format that has proven effective"
            )
        else:
            mined_pattern = "Apply successful patterns from similar tasks"

        # Extract patterns from examples
        pattern_descriptions = []
        for example in pattern_examples:
            if "pattern" in example:
                pattern_descriptions.append(
                    f"- {example['pattern']}: {example['input']} → {example['output']}"
                )

        patterns_text = (
            "\n".join(pattern_descriptions)
            if pattern_descriptions
            else "Using discovered effective patterns"
        )

        adaptive_instruction = ""
        if adaptive:
            adaptive_instruction = "\nAdapt these patterns as needed to fit the specific requirements of the current task."

        prompt = dedent_prompt(f"""
        Based on analysis of successful prompt patterns, I've identified effective approaches:

        Mined Patterns:
        {patterns_text}

        Key Pattern Focus: {mined_pattern}

        Task: {input_text}

        Apply the most relevant discovered patterns to solve this task effectively.{adaptive_instruction}

        Response using mined patterns:
        """)
        return prompt


class KNN(PromptTechnique):
    """
    KNN selects examples based on k-nearest neighbors.

    NOTE: This is currently a simplified implementation using random sampling.
    A full implementation would use text embeddings and semantic similarity.

    Future enhancements planned:
    1. Integrate with sentence-transformers for text embeddings
    2. Implement proper distance/similarity metrics (e.g., cosine similarity)
    3. Add caching for embeddings to improve performance
    4. Support different embedding models
    """

    def __init__(self):
        """Initialize KNN technique."""
        super().__init__(
            name="KNN",
            identifier="2.2.1.2",
            description="Selects examples using k-nearest neighbors approach (currently simplified).",
        )

    def generate_prompt(
        self,
        input_text: str,
        examples_pool: Optional[List[Dict[str, str]]] = None,
        k: int = 3,
        **kwargs,
    ) -> str:
        """
        Generate a few-shot prompt with KNN-selected examples.

        IMPORTANT: The current implementation uses random sampling instead of true KNN.
        For production use cases requiring accurate similarity matching, consider
        using a different technique or wait for future updates to this class.

        Args:
            input_text (str): Input text
            examples_pool (Optional[List[Dict[str, str]]]): Pool of available examples.
                Each example should have 'input' and 'output' keys.
                Defaults to empty list if None.
            k (int): Number of nearest neighbors to select (default: 3)
            **kwargs: Additional arguments

        Returns:
            str: Generated prompt with KNN-selected examples
        """
        # SIMPLIFICATION WARNING: This is not a true KNN implementation
        # In a proper implementation, we would:
        # 1. Compute embeddings for input_text and all examples in examples_pool
        # 2. Calculate semantic similarity (cosine similarity) between input and examples
        # 3. Select the k examples with highest similarity scores

        if examples_pool is None:
            examples_pool = []  # Initialize to empty list if None

        selected_examples = []
        if examples_pool:
            # For now, just randomly sample k examples from the pool
            selected_examples = random.sample(examples_pool, min(k, len(examples_pool)))

        # Format the selected examples
        if selected_examples:
            examples_text = "\n\n".join(
                [
                    f"Input: {example['input']}\nOutput: {example['output']}"
                    for example in selected_examples
                ]
            )
        else:
            examples_text = "[No similar examples found]"

        # Generate the prompt with the selected examples
        prompt = dedent_prompt(f"""
        Here are some examples that seem most relevant to your query:

        {examples_text}

        Now, for your query:
        Input: {input_text}
        Output:
        """)
        return prompt
