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
            **kwargs: Additional arguments:
                - num_steps (int): Number of solution steps to verify (default: 3)
                - verification_aspects (List[str]): Specific aspects to verify
                - domain (str): Optional domain context
                - verification_intensity (str): Level of verification ("basic", "thorough", "rigorous")
                - include_counterexamples (bool): Whether to actively seek counterexamples
            
        Returns:
            str: Generated Chain-of-Verification prompt
        """
        num_steps = kwargs.get("num_steps", 3)
        verification_aspects = kwargs.get("verification_aspects", [])
        domain = kwargs.get("domain", "")
        verification_intensity = kwargs.get("verification_intensity", "thorough")
        include_counterexamples = kwargs.get("include_counterexamples", False)
        
        domain_context = f" in the {domain} context" if domain else ""
        
        # Default verification aspects if none provided
        if not verification_aspects:
            verification_aspects = [
                "factual correctness",
                "logical consistency",
                "completeness of analysis",
                "appropriateness of methods used"
            ]
        
        verification_level = {
            "basic": "Check for obvious errors and inconsistencies.",
            "thorough": "Carefully examine assumptions, methods, and conclusions for validity and completeness.",
            "rigorous": "Conduct an exhaustive verification, challenging every aspect of the solution with alternative perspectives."
        }.get(verification_intensity, "Carefully examine assumptions, methods, and conclusions for validity and completeness.")
        
        counterexample_text = "\n- Actively attempt to construct counterexamples or cases where this solution might fail" if include_counterexamples else ""
        
        # Generate solution steps
        solution_steps = "\n".join([f"{i+1}. [Solution step {i+1}]" for i in range(num_steps)])
        
        # Generate verification sections
        verification_sections = ""
        for i in range(num_steps):
            step_num = i + 1
            previous_steps_ref = f" based on step{'s' if i > 0 else ''} {', '.join(str(j+1) for j in range(i))}" if i > 0 else ""
            
            verification_sections += f"""
        ## Verification of Step {step_num}:
        
        - **Original Step {step_num}:** [Restate the step to ensure clear focus]
        
        - **Verification Checklist:**
          - Are the assumptions valid{previous_steps_ref}? [Assess]
          - Is the approach appropriate? [Evaluate]
          - Are calculations/reasoning correct? [Verify]
          - Is the step addressing the right aspect of the problem? [Check]
        
        - **Critical Assessment:**
          - Potential issues or weaknesses: [Identify]
          - Alternative approaches to consider: [Suggest]{counterexample_text}
        
        - **Refinement:**
          - [Provide corrected/improved version of step {step_num}]
        
        """
        
        prompt = dedent_prompt(f"""
        # Problem Analysis and Self-Verification{domain_context}
        
        ## Problem Statement:
        {input_text}
        
        ## Verification Approach:
        I will first generate an initial solution, then critically verify each step using a {verification_intensity} approach. I will check specifically for issues with: {', '.join(verification_aspects)}.
        
        {verification_level}
        
        ## Initial Solution:
        {solution_steps}
        {verification_sections}
        ## Overall Verification:
        
        - **Integration Check:** [Verify that all steps work together coherently]
        - **Completeness Check:** [Ensure the solution addresses all aspects of the problem]
        - **Consistency Check:** [Confirm no contradictions between different steps]
        - **Reality Check:** [Assess whether the solution is reasonable given real-world constraints]
        
        ## Final Refined Solution:
        
        [Present the complete, verified solution with all corrections and improvements incorporated]
        
        ## Confidence Assessment:
        
        [Provide an assessment of confidence in the final solution, noting any remaining uncertainties]
        """)
        return prompt 