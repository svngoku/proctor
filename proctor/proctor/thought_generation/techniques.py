"""
Implementation of Thought Generation prompting techniques.
"""
from typing import List, Dict, Optional
from ..base import PromptTechnique
from ..utils import dedent_prompt

class ChainOfThought(PromptTechnique):
    """
    Chain-of-Thought (CoT) encourages step-by-step reasoning.
    """
    
    def __init__(self):
        """Initialize Chain-of-Thought technique."""
        super().__init__(
            name="Chain-of-Thought",
            identifier="2.2.3",
            description="Encourages step-by-step reasoning before giving an answer."
        )
    
    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Chain-of-Thought prompt.
        
        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
            
        Returns:
            str: Generated Chain-of-Thought prompt
        """
        # Note: This format guides the LLM to provide the steps.
        prompt = dedent_prompt(f"""
        {input_text}
        
        Let's work through this step-by-step:
        1. 
        2. 
        3. 
        
        Therefore, the answer is:
        """)
        return prompt


class ZeroShotCoT(PromptTechnique):
    """
    Zero-Shot Chain-of-Thought adds "Let's think step by step" to the prompt.
    """
    
    def __init__(self):
        """Initialize Zero-Shot Chain-of-Thought technique."""
        super().__init__(
            name="Zero-Shot CoT",
            identifier="2.2.3.1",
            description="Adds 'Let\'s think step by step\' to encourage reasoning."
        )
    
    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Zero-Shot Chain-of-Thought prompt.
        
        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
            
        Returns:
            str: Generated Zero-Shot CoT prompt
        """
        prompt = dedent_prompt(f"""
        {input_text}
        
        Let's think step by step.
        """)
        return prompt


class FewShotCoT(PromptTechnique):
    """
    Few-Shot Chain-of-Thought provides examples of reasoning.
    """
    
    def __init__(self):
        """Initialize Few-Shot Chain-of-Thought technique."""
        super().__init__(
            name="Few-Shot CoT",
            identifier="2.2.3.2",
            description="Provides examples of step-by-step reasoning."
        )
    
    def generate_prompt(
        self, 
        input_text: str, 
        examples: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Generate a Few-Shot Chain-of-Thought prompt.
        
        Args:
            input_text (str): Input text
            examples (Optional[List[Dict[str, str]]]): Examples with 'problem', 'reasoning', and 'answer' keys.
                                                     Defaults to predefined examples if None.
            **kwargs: Additional arguments
            
        Returns:
            str: Generated Few-Shot CoT prompt
        """
        if examples is None:
            examples = [
                {
                    "problem": "If John has 5 apples and gives 2 to Mary, how many does he have left?",
                    "reasoning": "John starts with 5 apples. He gives 2 apples to Mary. So he has 5 - 2 = 3 apples left.",
                    "answer": "3 apples"
                },
                {
                    "problem": "If a train travels 120 km in 2 hours, what is its speed?",
                    "reasoning": "Speed equals distance divided by time. The train covers 120 km in 2 hours. So its speed is 120 km / 2 hours = 60 km/hour.", # Corrected division symbol
                    "answer": "60 km/hour"
                }
            ]
        
        examples_text = "\n\n".join([
            f"Problem: {example['problem']}\n\nReasoning: {example['reasoning']}\n\nAnswer: {example['answer']}"
            for example in examples
        ])
        
        prompt = dedent_prompt(f"""
        Here are some examples of how to solve problems by reasoning step-by-step:

        {examples_text}

        Now, let's solve this problem using the same step-by-step approach:

        Problem: {input_text}

        Reasoning:
        """)
        return prompt 