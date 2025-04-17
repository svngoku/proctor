from typing import List, Dict, Optional
from ..base import PromptTechnique
from ..utils import dedent_prompt

class ChainOfThought(PromptTechnique):
    """
    Chain-of-Thought (CoT) encourages step-by-step reasoning for problem-solving.
    """
    
    def __init__(self, num_steps: int = 3):
        """
        Initialize Chain-of-Thought technique.

        Args:
            num_steps (int): Number of reasoning steps in the prompt. Defaults to 3.
        """
        super().__init__(
            name="Chain-of-Thought",
            identifier="2.2.3",
            description="Encourages step-by-step reasoning before providing an answer."
        )
        self.num_steps = max(1, num_steps)  # Ensure at least one step
    
    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Chain-of-Thought prompt with step-by-step reasoning guidance.

        Args:
            input_text (str): The problem or question to be solved.
            **kwargs: Additional arguments (e.g., custom instructions).

        Returns:
            str: Formatted Chain-of-Thought prompt.

        Raises:
            ValueError: If input_text is empty or invalid.
        """
        if not input_text or not isinstance(input_text, str):
            raise ValueError("input_text must be a non-empty string")
        
        # Generate numbered steps dynamically based on num_steps
        steps = "\n".join(f"{i+1}. " for i in range(self.num_steps))
        custom_instructions = kwargs.get("custom_instructions", "Let's work through this step-by-step:")
        
        prompt = dedent_prompt(f"""
        {input_text}

        {custom_instructions}
        {steps}
        
        Therefore, the final answer is:
        """)
        return prompt


class ZeroShotCoT(PromptTechnique):
    """
    Zero-Shot Chain-of-Thought adds a reasoning prompt without examples.
    """
    
    def __init__(self):
        """Initialize Zero-Shot Chain-of-Thought technique."""
        super().__init__(
            name="Zero-Shot CoT",
            identifier="2.2.3.1",
            description="Encourages step-by-step reasoning with a simple prompt."
        )
    
    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Zero-Shot Chain-of-Thought prompt.

        Args:
            input_text (str): The problem or question to be solved.
            **kwargs: Additional arguments (e.g., custom instructions).

        Returns:
            str: Formatted Zero-Shot CoT prompt.

        Raises:
            ValueError: If input_text is empty or invalid.
        """
        if not input_text or not isinstance(input_text, str):
            raise ValueError("input_text must be a non-empty string")
        
        custom_instructions = kwargs.get("custom_instructions", "Let's think step by step to solve this:")
        
        prompt = dedent_prompt(f"""
        {input_text}

        {custom_instructions}
        """)
        return prompt


class FewShotCoT(PromptTechnique):
    """
    Few-Shot Chain-of-Thought provides examples of step-by-step reasoning.
    """
    
    def __init__(self):
        """Initialize Few-Shot Chain-of-Thought technique."""
        super().__init__(
            name="Few-Shot CoT",
            identifier="2.2.3.2",
            description="Provides examples of step-by-step reasoning to guide problem-solving."
        )
    
    def _validate_examples(self, examples: List[Dict[str, str]]) -> None:
        """
        Validate the structure and content of provided examples.

        Args:
            examples (List[Dict[str, str]]): List of example dictionaries.

        Raises:
            ValueError: If examples are invalid or missing required keys.
        """
        required_keys = {"problem", "reasoning", "answer"}
        for example in examples:
            if not isinstance(example, dict):
                raise ValueError("Each example must be a dictionary")
            if not all(key in example for key in required_keys):
                raise ValueError(f"Each example must contain {required_keys}")
            if not all(isinstance(example[key], str) and example[key] for key in required_keys):
                raise ValueError("Example fields must be non-empty strings")

    def generate_prompt(
        self, 
        input_text: str, 
        examples: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Generate a Few-Sh部分

        Args:
            input_text (str): The problem or question to be solved.
            examples (Optional[List[Dict[str, str]]]): Examples with 'problem', 'reasoning', and 'answer' keys.
            **kwargs: Additional arguments (e.g., custom instructions).

        Returns:
            str: Formatted Few-Shot CoT prompt.

        Raises:
            ValueError: If input_text is empty or examples are invalid.
        """
        if not input_text or not isinstance(input_text, str):
            raise ValueError("input_text must be a non-empty string")

        # Use default examples if none provided
        if examples is None:
            examples = [
                {
                    "problem": "If John has 5 apples and gives 2 to Mary, how many does he have left?",
                    "reasoning": "John starts with 5 apples. He gives 2 apples to Mary. Subtracting the apples given away, 5 - 2 = 3 apples remain.",
                    "answer": "3 apples"
                },
                {
                    "problem": "If a train travels 120 km in 2 hours, what is its speed?",
                    "reasoning": "To find speed, use the formula: speed = distance ÷ time. The train travels 120 km in 2 hours. Therefore, speed = 120 km ÷ 2 hours = 60 km/hour.",
                    "answer": "60 km/hour"
                }
            ]
        
        # Validate examples
        self._validate_examples(examples)
        
        # Format examples into prompt
        examples_text = "\n\n".join([
            f"Problem: {example['problem']}\n\nReasoning: {example['reasoning']}\n\nAnswer: {example['answer']}"
            for example in examples
        ])
        
        custom_instructions = kwargs.get("custom_instructions", "Use the same step-by-step reasoning approach as shown in the examples to solve the following problem:")
        
        prompt = dedent_prompt(f"""
        Below are examples of solving problems with step-by-step reasoning:

        {examples_text}

        {custom_instructions}

        Problem: {input_text}

        Reasoning:
        """)
        return prompt