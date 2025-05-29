"""
Implementation of Zero-Shot prompting techniques.
"""
from ..base import PromptTechnique
from ..utils import dedent_prompt

class EmotionPrompting(PromptTechnique):
    """
    Emotion Prompting incorporates emotional cues to guide the model.
    
    This technique directs the model to approach a task with a specific emotion,
    which can influence the tone, style, and framing of its response.
    """
    
    def __init__(self):
        """Initialize Emotion Prompting technique."""
        super().__init__(
            name="Emotion Prompting",
            identifier="2.2.2.1", # Unique identifier within zero-shot techniques
            description="Incorporates emotional cues in prompts to guide responses."
        )
    
    def generate_prompt(self, input_text: str, emotion: str = "excited", **kwargs) -> str:
        """
        Generate an emotion-based prompt.
        
        Args:
            input_text (str): Input text
            emotion (str): Emotion to incorporate (default: "excited")
            **kwargs: Additional arguments
                - context (str): Optional background context to help frame the task
                - intensity (str): Optional intensity level (e.g., "very", "somewhat")
            
        Returns:
            str: Generated prompt with emotional cues
        """
        context = kwargs.get("context", "")
        intensity = kwargs.get("intensity", "")
        intensity_phrase = f"{intensity} {emotion}" if intensity else emotion
        
        prompt = dedent_prompt(f"""
        As an AI assistant, I want you to respond with {intensity_phrase} energy to this task.
        
        {context}
        
        Task: {input_text}
        
        When responding:
        - Express genuine {emotion} about this topic
        - Use language that conveys {emotion} (tone, word choice, pacing)
        - Maintain this emotional perspective throughout your response
        - Still prioritize accuracy and helpfulness
        
        Begin your response now, showing your {emotion} perspective:
        """)
        return prompt


class RolePrompting(PromptTechnique):
    """
    Role Prompting assigns a specific role to the model.
    
    This technique instructs the model to adopt a particular persona or role
    (e.g., "expert", "teacher", "doctor") when responding to the input,
    which can influence the perspective, depth, and style of its answer.
    """
    
    def __init__(self):
        """Initialize Role Prompting technique."""
        super().__init__(
            name="Role Prompting",
            identifier="2.2.2.2", # Unique identifier within zero-shot techniques
            description="Assigns a specific role to the model to guide its responses."
        )
    
    def generate_prompt(self, input_text: str, role: str = "expert", **kwargs) -> str:
        """
        Generate a role-based prompt.
        
        Args:
            input_text (str): Input text
            role (str): Role to assign to the model
            **kwargs: Additional arguments
                - field (str): Optional specific field/domain of expertise
                - experience (str): Optional experience level (e.g., "senior", "world-renowned")
                - audience (str): Optional target audience for the response
            
        Returns:
            str: Generated prompt with role assignment
        """
        field = kwargs.get("field", "this field")
        experience = kwargs.get("experience", "")
        audience = kwargs.get("audience", "")
        
        audience_str = f"Your target audience is {audience}. " if audience else ""
        experience_role = f"{experience} {role}" if experience else role
        
        prompt = dedent_prompt(f"""
        I want you to assume the role of a {experience_role} in {field}. Think about the knowledge, perspective, and communication style that a real {role} would have.
        
        {audience_str}Given your expertise as a {role}, please address the following:
        
        {input_text}
        
        When responding:
        - Use terminology, concepts, and frameworks common in {field}
        - Apply the analytical approach typical of a {role}
        - Structure your response as a {role} would in a professional context
        - Draw on specialized knowledge available to someone in this role
        - Maintain this perspective throughout your entire response
        
        Your response as a {role}:
        """)
        return prompt


class SelfAsk(PromptTechnique):
    """
    Self-Ask encourages the model to ask and answer its own questions.
    
    This technique structures the prompt to guide the model through a
    self-questioning process, where it identifies relevant sub-questions,
    answers them, and then uses those answers to formulate a final response.
    This can lead to more thorough analysis and reasoning.
    """
    
    def __init__(self):
        """Initialize Self-Ask technique."""
        super().__init__(
            name="Self-Ask",
            identifier="2.2.2.3", # Unique identifier within zero-shot techniques
            description="Prompts the model to ask and answer its own questions."
        )
    
    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a self-ask prompt.
        
        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - num_questions (int): Optional number of follow-up questions to ask (default: 3)
                - depth (str): Optional depth of analysis ("shallow", "moderate", "deep")
                - domain (str): Optional specific domain to focus questions in
            
        Returns:
            str: Generated self-ask prompt
        """
        num_questions = kwargs.get("num_questions", 3)
        depth = kwargs.get("depth", "moderate")
        domain = kwargs.get("domain", "")
        
        domain_str = f" in the domain of {domain}" if domain else ""
        
        depth_guidance = {
            "shallow": "focus on basic clarifications and direct implications",
            "moderate": "explore key factors, important connections, and significant implications",
            "deep": "delve into underlying principles, complex interconnections, and explore nuanced aspects"
        }.get(depth, "explore key factors, important connections, and significant implications")
        
        questions = "\n\n".join([
            f"{i+1}. [Ask a specific, focused question that helps address an important aspect of the main question{domain_str}]\n[Provide a clear, evidence-based answer to this question]"
            for i in range(num_questions)
        ])
        
        prompt = dedent_prompt(f"""
        Main Question: {input_text}
        
        To thoroughly answer this question, I'll use a self-questioning approach. I'll identify and answer {num_questions} key follow-up questions that will help me build toward a comprehensive response. For each question, I'll {depth_guidance}.

        {questions}
        
        Now, synthesizing all the information from my self-questioning process:
        
        Final comprehensive answer to the original question:
        """)
        return prompt 