"""
Implementation of Self-Criticism prompting techniques.
"""
from ..base import PromptTechnique
from ..utils import dedent_prompt

class ChainOfVerification(PromptTechnique):
    """
    Chain-of-Verification reviews and verifies reasoning.
    (Note: Guides LLM to perform verification steps)
    """
    
    def __init__(self):
        """Initialize Chain-of-Verification technique."""
        super().__init__(
            name="Chain-of-Verification",
            identifier="2.2.6",
            description="Reviews and verifies each step of reasoning."
        )
    
    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Chain-of-Verification prompt.
        
        Args:
            input_text (str): Input text (representing the problem)
            **kwargs: Additional arguments
            
        Returns:
            str: Generated Chain-of-Verification prompt
        """
        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        Generate an initial plan or solution approach:
        1. [Step 1]
        2. [Step 2]
        3. [Step 3]
        
        Now, review and verify each step of the plan:
        
        Verification of Step 1:
        - Is the understanding/assumption correct? [Check]
        - Is the calculation/reasoning sound? [Check]
        - Identify potential errors or necessary adjustments: [Analysis/Correction]
        
        Verification of Step 2:
        - Is the understanding/assumption correct based on Step 1? [Check]
        - Is the calculation/reasoning sound? [Check]
        - Identify potential errors or necessary adjustments: [Analysis/Correction]
        
        Verification of Step 3:
        - Is the understanding/assumption correct based on previous steps? [Check]
        - Is the calculation/reasoning sound? [Check]
        - Identify potential errors or necessary adjustments: [Analysis/Correction]
        
        Based on the verification, provide the final, refined solution:
        [Corrected and verified solution]
        """)
        return prompt 