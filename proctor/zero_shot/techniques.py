"""
Implementation of Zero-Shot prompting techniques.
"""
from ..base import PromptTechnique
from ..utils import dedent_prompt

class EmotionPrompting(PromptTechnique):
    """
    Emotion Prompting incorporates emotional cues to guide the model.
    """
    
    def __init__(self):
        """Initialize Emotion Prompting technique."""
        super().__init__(
            name="Emotion Prompting",
            identifier="2.2.2", # Note: Identifier seems repeated, might need clarification from source
            description="Incorporates emotional cues in prompts to guide responses."
        )
    
    def generate_prompt(self, input_text: str, emotion: str = "excited", **kwargs) -> str:
        """
        Generate an emotion-based prompt.
        
        Args:
            input_text (str): Input text
            emotion (str): Emotion to incorporate (default: "excited")
            **kwargs: Additional arguments
            
        Returns:
            str: Generated prompt with emotional cues
        """
        prompt = dedent_prompt(f"""
        I want you to feel {emotion} about helping me with this task.
        
        Task: {input_text}
        
        Please approach this with {emotion} energy and enthusiasm.
        """)
        return prompt


class RolePrompting(PromptTechnique):
    """
    Role Prompting assigns a specific role to the model.
    """
    
    def __init__(self):
        """Initialize Role Prompting technique."""
        super().__init__(
            name="Role Prompting",
            identifier="2.2.2", # Note: Identifier seems repeated
            description="Assigns a specific role to the model to guide its responses."
        )
    
    def generate_prompt(self, input_text: str, role: str = "expert", **kwargs) -> str:
        """
        Generate a role-based prompt.
        
        Args:
            input_text (str): Input text
            role (str): Role to assign to the model
            **kwargs: Additional arguments
            
        Returns:
            str: Generated prompt with role assignment
        """
        prompt = dedent_prompt(f"""
        I want you to act as a {role} in this field. 
        
        {input_text}
        
        Please respond as a {role} would.
        """)
        return prompt


class SelfAsk(PromptTechnique):
    """
    Self-Ask encourages the model to ask and answer its own questions.
    """
    
    def __init__(self):
        """Initialize Self-Ask technique."""
        super().__init__(
            name="Self-Ask",
            identifier="2.2.2", # Note: Identifier seems repeated
            description="Prompts the model to ask and answer its own questions."
        )
    
    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a self-ask prompt.
        
        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
            
        Returns:
            str: Generated self-ask prompt
        """
        # Note: The prompt format suggests the LLM fills in the questions/answers
        prompt = dedent_prompt(f"""
        Question: {input_text}
        
        To answer this question, I'll need to ask myself some follow-up questions:
        1. [Ask a relevant question]
        [Answer that question]
        
        2. [Ask another relevant question]
        [Answer that question]
        
        Based on these answers, the final answer is:
        """)
        return prompt 