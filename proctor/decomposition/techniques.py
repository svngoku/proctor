"""
Implementation of Decomposition prompting techniques.
"""
from ..base import PromptTechnique
from ..utils import dedent_prompt

class DECOMP(PromptTechnique):
    """
    DECOMP breaks down complex problems into simpler subproblems.
    (Note: Guides the LLM to perform decomposition)
    """
    
    def __init__(self):
        """Initialize DECOMP technique."""
        super().__init__(
            name="DECOMP",
            identifier="2.2.4",
            description="Breaks down complex problems into simpler subproblems."
        )
    
    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a DECOMP prompt.
        
        Args:
            input_text (str): Input text (the complex problem)
            **kwargs: Additional arguments
            
        Returns:
            str: Generated DECOMP prompt
        """
        prompt = dedent_prompt(f"""
        Complex Problem: {input_text}
        
        Let's break this down into simpler, manageable subproblems:
        
        Subproblem 1: [Identify and describe the first subproblem]
        Solution to Subproblem 1:
        [Solve the first subproblem]
        
        Subproblem 2: [Identify and describe the second subproblem, possibly depending on the first]
        Solution to Subproblem 2:
        [Solve the second subproblem]
        
        Subproblem 3: [Identify and describe the third subproblem... continue as needed]
        Solution to Subproblem 3:
        [Solve the third subproblem]
        
        Now, combine the solutions to the subproblems to address the original complex problem:
        [Synthesis of subproblem solutions]
        
        Therefore, the final answer to the original complex problem is:
        [Final answer]
        """)
        return prompt 