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
            **kwargs: Additional arguments:
                - num_subproblems (int): Number of subproblems to identify (default: 3)
                - approach (str): Decomposition approach ("sequential", "parallel", "hierarchical")
                - domain (str): Optional specific domain context
                - clear_dependencies (bool): Whether to explicitly track dependencies between subproblems
            
        Returns:
            str: Generated DECOMP prompt
        """
        num_subproblems = kwargs.get("num_subproblems", 3)
        approach = kwargs.get("approach", "sequential")
        domain = kwargs.get("domain", "")
        clear_dependencies = kwargs.get("clear_dependencies", False)
        
        domain_context = f" in the {domain} domain" if domain else ""
        
        approach_guidance = {
            "sequential": "Break the problem down into sequential steps, where each subproblem builds on the previous one.",
            "parallel": "Identify independent aspects of the problem that can be solved separately and then combined.",
            "hierarchical": "Break the problem into major components, then further decompose each component as needed."
        }.get(approach, "Break the problem down into manageable parts that are easier to solve individually.")
        
        dependencies_text = "\n- Explicitly note how each subproblem depends on or relates to others" if clear_dependencies else ""
        
        # Generate subproblems dynamically
        subproblems = ""
        for i in range(num_subproblems):
            subproblems += f"""
        Subproblem {i+1}: [Identify and precisely describe a clear, specific aspect of the main problem]
        - Why this subproblem is important: [Explain why solving this contributes to the overall solution]
        - Key information needed: [Identify what data/concepts are needed to solve this part]
        
        Solution to Subproblem {i+1}:
        [Solve this subproblem with clear, systematic reasoning]
        
        """
        
        prompt = dedent_prompt(f"""
        # Complex Problem Analysis{domain_context}:
        
        Problem Statement: {input_text}
        
        ## Decomposition Strategy:
        {approach_guidance}{dependencies_text}
        
        ## Breaking Down the Problem:
        {subproblems}
        ## Integration and Synthesis:
        [Explain how the solutions to subproblems connect and build toward the complete solution]
        [Identify any important insights that emerge from combining the partial solutions]
        [Address any remaining aspects not covered by the individual subproblems]
        
        ## Final Comprehensive Solution:
        [Provide the complete, integrated solution to the original problem]
        
        ## Verification:
        [Verify that the solution addresses all aspects of the original problem]
        [Check for consistency and correctness across subproblem solutions]
        """)
        return prompt 