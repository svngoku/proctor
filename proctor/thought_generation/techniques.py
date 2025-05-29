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
            **kwargs: Additional arguments:
                - custom_instructions (str): Custom instruction text
                - approach (str): Optional reasoning approach (e.g., "analytical", "mathematical")
                - detail_level (str): Optional level of detail ("brief", "standard", "detailed")
                - include_alternatives (bool): Whether to explore alternative approaches

        Returns:
            str: Formatted Chain-of-Thought prompt.

        Raises:
            ValueError: If input_text is empty or invalid.
        """
        if not input_text or not isinstance(input_text, str):
            raise ValueError("input_text must be a non-empty string")
        
        approach = kwargs.get("approach", "")
        detail_level = kwargs.get("detail_level", "standard")
        include_alternatives = kwargs.get("include_alternatives", False)
        
        # Customize guidance based on parameters
        approach_text = f" using a {approach} approach" if approach else ""
        
        detail_guidance = {
            "brief": "Focus on key insights with minimal explanation.",
            "standard": "Provide balanced reasoning with moderate detail.",
            "detailed": "Explore nuances and provide comprehensive explanation."
        }.get(detail_level, "Provide balanced reasoning with moderate detail.")
        
        alternatives_text = "\n\nConsider at least one alternative approach or perspective before reaching your final conclusion." if include_alternatives else ""
        
        custom_instructions = kwargs.get(
            "custom_instructions", 
            f"Let's work through this{approach_text} step-by-step. {detail_guidance}{alternatives_text}"
        )
        
        # Create a structured prompt with clearer guidance for each step
        steps_text = ""
        for i in range(self.num_steps):
            step_num = i + 1
            if step_num == 1:
                steps_text += f"{step_num}. [Identify the key components of the problem]\n\n"
            elif step_num == self.num_steps:
                steps_text += f"{step_num}. [Derive the final result based on previous steps]\n\n"
            else:
                steps_text += f"{step_num}. [Apply logical reasoning to continue from previous steps]\n\n"
        
        prompt = dedent_prompt(f"""
        Problem/Question: {input_text}

        {custom_instructions}
        
        {steps_text}Therefore, the final answer is:
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
            **kwargs: Additional arguments:
                - custom_instructions (str): Custom instruction text
                - domain (str): Optional domain for contextualizing the reasoning
                - reasoning_style (str): Optional reasoning style ("analytical", "creative", etc.)
                - complexity (str): Optional complexity level ("simple", "intermediate", "advanced")

        Returns:
            str: Formatted Zero-Shot CoT prompt.

        Raises:
            ValueError: If input_text is empty or invalid.
        """
        if not input_text or not isinstance(input_text, str):
            raise ValueError("input_text must be a non-empty string")
        
        domain = kwargs.get("domain", "")
        reasoning_style = kwargs.get("reasoning_style", "")
        complexity = kwargs.get("complexity", "")
        
        domain_context = f" in the domain of {domain}" if domain else ""
        style_context = f" using {reasoning_style} reasoning" if reasoning_style else ""
        complexity_guidance = {
            "simple": "Break this down into basic steps, focusing on the core concepts.",
            "intermediate": "Analyze this methodically, considering important factors and relationships.",
            "advanced": "Examine this comprehensively, addressing nuances and exploring deeper implications."
        }.get(complexity, "")
        
        complexity_text = f" {complexity_guidance}" if complexity else ""
        
        custom_instruction_base = f"Let's think step by step{domain_context}{style_context} to solve this problem:{complexity_text}"
        custom_instructions = kwargs.get("custom_instructions", custom_instruction_base)
        
        prompt = dedent_prompt(f"""
        Problem/Question: {input_text}

        {custom_instructions}

        1. [First, I'll identify what the problem is asking and key information provided]
        
        2. [Next, I'll determine an approach to solve this systematically]
        
        3. [I'll work through each logical step of my solution]
        
        4. [Finally, I'll verify my solution and formulate my answer]
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
        Generate a Few-Shot Chain-of-Thought prompt with examples of reasoning.

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
        
        domain = kwargs.get("domain", "")
        focus_areas = kwargs.get("focus_areas", [])
        
        domain_text = f" in {domain}" if domain else ""
        focus_text = ""
        if focus_areas:
            focus_text = "\n- Pay special attention to: " + ", ".join(focus_areas)
        
        custom_instructions = kwargs.get(
            "custom_instructions",
            f"Use the same step-by-step reasoning approach as shown in the examples to solve the following problem{domain_text}:"
        )
        
        prompt = dedent_prompt(f"""
        Below are examples of problems solved using effective step-by-step reasoning. Study these patterns carefully:

        {examples_text}

        {custom_instructions}
        {focus_text}

        Problem: {input_text}

        I'll solve this by following a similar reasoning process:
        1. First, I'll understand what the problem is asking
        2. Then, I'll identify the key information and constraints
        3. Next, I'll apply a systematic approach similar to the examples
        4. Finally, I'll derive the answer through careful reasoning

        Reasoning:
        """)
        return prompt