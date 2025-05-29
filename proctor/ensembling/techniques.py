"""
Implementation of Ensembling prompting techniques.
"""
from typing import List
from ..base import PromptTechnique
from ..utils import dedent_prompt

class SelfConsistency(PromptTechnique):
    """
    Self-Consistency generates multiple reasoning paths and finds consensus.
    (Note: Prompt generation guides the LLM; actual analysis of paths is complex)
    """
    
    def __init__(self):
        """Initialize Self-Consistency technique."""
        super().__init__(
            name="Self-Consistency",
            identifier="2.2.5",
            description="Generates multiple reasoning paths and finds consensus."
        )
    
    def generate_prompt(
        self, 
        input_text: str,
        num_paths: int = 3,
        **kwargs
    ) -> str:
        """
        Generate a Self-Consistency prompt.
        
        Args:
            input_text (str): Input text
            num_paths (int): Number of reasoning paths to generate (default: 3)
            **kwargs: Additional arguments:
                - approach_diversity (bool): Whether to ensure diversity in approaches
                - domain (str): Optional domain context
                - include_metacognition (bool): Whether to include reflection on thinking process
                - reasoning_styles (List[str]): Optional list of specific reasoning styles to use
                - path_length (str): Guidance on path length ("brief", "standard", "detailed")
            
        Returns:
            str: Generated Self-Consistency prompt
        """
        approach_diversity = kwargs.get("approach_diversity", True)
        domain = kwargs.get("domain", "")
        include_metacognition = kwargs.get("include_metacognition", False)
        reasoning_styles = kwargs.get("reasoning_styles", [])
        path_length = kwargs.get("path_length", "standard")
        
        domain_context = f" in the {domain} domain" if domain else ""
        
        # Generate guidance for path length
        length_guidance = {
            "brief": "Keep each reasoning path concise, focusing on key insights.",
            "standard": "Provide a balanced level of detail in each reasoning path.",
            "detailed": "Elaborate thoroughly on each reasoning path, exploring nuances."
        }.get(path_length, "Provide a balanced level of detail in each reasoning path.")
        
        # Create diversity instructions if enabled
        diversity_instructions = """
        - Ensure each path uses a substantially different approach or perspective
        - Avoid simply rephrasing the same reasoning with minor variations
        """ if approach_diversity else ""
        
        # Create metacognition instructions if enabled
        metacognition = """
        - For each path, briefly note your confidence level and any uncertainties
        - Identify which aspects of the problem were most challenging in each path
        """ if include_metacognition else ""
        
        # Use specified reasoning styles if provided
        styles_text = ""
        if reasoning_styles:
            styles_list = "\n".join([f"  - Path {i+1}: Use {style} reasoning" for i, style in enumerate(reasoning_styles[:num_paths])])
            styles_text = f"\nI will use different reasoning styles for each path:\n{styles_list}\n"
        
        # Generate paths with more guidance
        paths = ""
        for i in range(num_paths):
            style_note = f" using {reasoning_styles[i]} reasoning" if reasoning_styles and i < len(reasoning_styles) else ""
            
            paths += f"""
        Path {i+1}{style_note}:
        [Start with a distinct approach to the problem]
        [Develop this approach step by step with clear reasoning]
        [Maintain logical consistency throughout this path]
        [Draw a specific conclusion based solely on this path's reasoning]
        
        Conclusion {i+1}: [Specific answer derived from path {i+1}]
        
        """
        
        prompt = dedent_prompt(f"""
        # Multiple-Path Problem Solving{domain_context}
        
        ## Problem Statement:
        {input_text}
        
        ## Approach:
        I will solve this problem through {num_paths} independent reasoning paths. {length_guidance}
        {styles_text}
        Key guidelines for this multi-path analysis:
        {diversity_instructions}{metacognition}
        
        ## Independent Reasoning Paths:
        {paths}
        ## Analysis of Results:
        
        - **Summary of Conclusions:**
          [List each conclusion with a count if any are identical]
        
        - **Comparative Analysis:**
          [Analyze the similarities and differences between the paths]
          [Identify strengths and weaknesses of each approach]
          [Note which aspects of the problem were consistently addressed across paths]
        
        - **Confidence Assessment:**
          [Evaluate relative confidence in each path based on rigor, completeness, and logical soundness]
        
        ## Consensus Determination:
        
        Based on the multiple reasoning paths, the most reliable conclusion is:
        [Provide the final answer with justification for why it represents the best consensus]
        
        ## Reflection on Multi-Path Approach:
        [Briefly note how using multiple paths improved the solution compared to a single-path approach]
        """)
        return prompt 