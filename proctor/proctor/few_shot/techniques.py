"""
Implementation of Few-Shot prompting techniques.
"""
from typing import List, Dict, Optional
import random # Added for KNN example
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
            description="Generates examples for few-shot learning."
        )
    
    def generate_prompt(self, input_text: str, examples: Optional[List[Dict[str, str]]] = None, **kwargs) -> str:
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
                {"input": "Example input 3", "output": "Example output 3"}
            ]
        
        examples_text = "\n\n".join([
            f"Input: {example['input']}\nOutput: {example['output']}"
            for example in examples
        ])
        
        prompt = dedent_prompt(f"""
        I'll show you some examples of how to solve this type of problem:

        {examples_text}

        Now, please solve the following:
        Input: {input_text}
        Output:
        """)
        return prompt


class KNN(PromptTechnique):
    """
    KNN selects examples based on k-nearest neighbors.
    (Simplified implementation using random sampling)
    """
    
    def __init__(self):
        """Initialize KNN technique."""
        super().__init__(
            name="KNN",
            identifier="2.2.1.2",
            description="Selects examples using k-nearest neighbors approach (simplified)."
        )
    
    def generate_prompt(
        self, 
        input_text: str, 
        examples_pool: Optional[List[Dict[str, str]]] = None,
        k: int = 3,
        **kwargs
    ) -> str:
        """
        Generate a few-shot prompt with KNN-selected examples.
        
        Args:
            input_text (str): Input text
            examples_pool (Optional[List[Dict[str, str]]]): Pool of available examples.
                                                         Defaults to empty list if None.
            k (int): Number of nearest neighbors to select (default: 3)
            **kwargs: Additional arguments
            
        Returns:
            str: Generated prompt with KNN-selected examples
        """
        # In a real implementation, we'd compute embeddings and find nearest neighbors
        # For this simplified version, we'll just select k random examples
        if examples_pool is None:
            examples_pool = [] # Initialize to empty list if None
            
        selected_examples = []
        if examples_pool:
            selected_examples = random.sample(examples_pool, min(k, len(examples_pool)))
        
        examples_text = "\n\n".join([
            f"Input: {example['input']}\nOutput: {example['output']}"
            for example in selected_examples
        ]) if selected_examples else "[No similar examples found]"
        
        prompt = dedent_prompt(f"""
        Here are some examples that seem most relevant to your query:

        {examples_text}

        Now, for your query:
        Input: {input_text}
        Output:
        """)
        return prompt 