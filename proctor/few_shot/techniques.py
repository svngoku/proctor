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
            description="Selects examples using k-nearest neighbors approach (currently simplified)."
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
            examples_pool = [] # Initialize to empty list if None
            
        selected_examples = []
        if examples_pool:
            # For now, just randomly sample k examples from the pool
            selected_examples = random.sample(examples_pool, min(k, len(examples_pool)))
        
        # Format the selected examples
        if selected_examples:
            examples_text = "\n\n".join([
                f"Input: {example['input']}\nOutput: {example['output']}"
                for example in selected_examples
            ])
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