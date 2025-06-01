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
            identifier="2.2.2.1",  # Unique identifier within zero-shot techniques
            description="Incorporates emotional cues in prompts to guide responses.",
        )

    def generate_prompt(
        self, input_text: str, emotion: str = "excited", **kwargs
    ) -> str:
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
            identifier="2.2.2.2",  # Unique identifier within zero-shot techniques
            description="Assigns a specific role to the model to guide its responses.",
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


class StylePrompting(PromptTechnique):
    """
    Style Prompting guides the model to respond in a specific writing or communication style.

    This technique instructs the model to adopt particular stylistic characteristics
    such as formal/informal tone, specific writing styles, or communication patterns.
    """

    def __init__(self):
        """Initialize Style Prompting technique."""
        super().__init__(
            name="Style Prompting",
            identifier="2.2.2.3",
            description="Guides the model to respond in a specific writing or communication style.",
        )

    def generate_prompt(
        self, input_text: str, style: str = "professional", **kwargs
    ) -> str:
        """
        Generate a style-guided prompt.

        Args:
            input_text (str): Input text
            style (str): Style to adopt (e.g., "professional", "casual", "academic", "creative")
            **kwargs: Additional arguments
                - tone (str): Optional tone specification
                - format (str): Optional format specification
                - audience (str): Optional target audience

        Returns:
            str: Generated prompt with style guidance
        """
        tone = kwargs.get("tone", "")
        format_spec = kwargs.get("format", "")
        audience = kwargs.get("audience", "")

        tone_str = f" with a {tone} tone" if tone else ""
        format_str = f" in {format_spec} format" if format_spec else ""
        audience_str = f" for {audience}" if audience else ""

        prompt = dedent_prompt(f"""
        I want you to respond to the following in a {style} style{tone_str}{format_str}{audience_str}.
        
        Task: {input_text}
        
        Style Guidelines:
        - Adopt the characteristic features of {style} writing
        - Use appropriate vocabulary and sentence structure for this style
        - Maintain consistency in your stylistic choices throughout
        - Ensure the content remains accurate and helpful while following the style
        
        Respond now in the requested {style} style:
        """)
        return prompt


class S2A(PromptTechnique):
    """
    System 2 Attention (S2A) prompting technique.

    This technique encourages deliberate, careful reasoning by prompting the model
    to engage its "System 2" thinking - slow, deliberate, and analytical processing.
    """

    def __init__(self):
        """Initialize S2A technique."""
        super().__init__(
            name="System 2 Attention (S2A)",
            identifier="2.2.2.4",
            description="Encourages deliberate, analytical reasoning through System 2 thinking.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a System 2 Attention prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - focus_areas (list): Optional specific areas to focus attention on
                - analysis_depth (str): Optional depth level ("basic", "detailed", "comprehensive")

        Returns:
            str: Generated S2A prompt
        """
        focus_areas = kwargs.get("focus_areas", [])
        analysis_depth = kwargs.get("analysis_depth", "detailed")

        focus_str = ""
        if focus_areas:
            focus_str = f"\nPay particular attention to: {', '.join(focus_areas)}"

        depth_guidance = {
            "basic": "Take time to consider the key aspects",
            "detailed": "Carefully analyze multiple dimensions and implications",
            "comprehensive": "Thoroughly examine all relevant factors, potential biases, and complex interactions",
        }.get(analysis_depth, "Carefully analyze multiple dimensions and implications")

        prompt = dedent_prompt(f"""
        I want you to engage in slow, deliberate, and careful thinking about this task. 
        
        {depth_guidance}. Avoid quick, automatic responses and instead:
        
        1. Take a moment to understand what's being asked
        2. Consider multiple perspectives and potential approaches
        3. Identify any assumptions or biases that might affect your reasoning
        4. Think through the implications of different aspects
        5. Synthesize your careful analysis into a well-reasoned response
        
        Task: {input_text}{focus_str}
        
        Now, engage your deliberate, analytical thinking to respond:
        """)
        return prompt


class SimToM(PromptTechnique):
    """
    Simulation Theory of Mind (SimToM) prompting technique.

    This technique prompts the model to simulate different perspectives and mental states
    to better understand and respond to tasks involving human behavior, motivations, or social dynamics.
    """

    def __init__(self):
        """Initialize SimToM technique."""
        super().__init__(
            name="Simulation Theory of Mind (SimToM)",
            identifier="2.2.2.5",
            description="Simulates different perspectives and mental states for better understanding.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a SimToM prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - perspectives (list): Optional specific perspectives to consider
                - context (str): Optional social/emotional context
                - depth (str): Optional depth of perspective-taking

        Returns:
            str: Generated SimToM prompt
        """
        perspectives = kwargs.get(
            "perspectives", ["different stakeholders", "various viewpoints"]
        )
        context = kwargs.get("context", "")
        depth = kwargs.get("depth", "moderate")

        context_str = f"\nContext: {context}" if context else ""
        perspectives_str = ", ".join(perspectives)

        depth_guidance = {
            "basic": "consider what others might think or feel",
            "moderate": "deeply consider the mental states, motivations, and perspectives of others",
            "advanced": "comprehensively simulate the complex mental models, emotional states, and reasoning patterns of different individuals",
        }.get(
            depth,
            "deeply consider the mental states, motivations, and perspectives of others",
        )

        prompt = dedent_prompt(f"""
        For this task, I want you to {depth_guidance} involved in or affected by this situation.
        
        Task: {input_text}{context_str}
        
        Before responding, mentally simulate:
        1. What different people might be thinking and feeling
        2. What motivations and goals various parties might have
        3. How different perspectives might interpret this situation
        4. What concerns, hopes, or expectations others might hold
        5. How social and emotional factors might influence the situation
        
        Consider perspectives from: {perspectives_str}
        
        Now, using your simulation of these different mental states and perspectives, provide a thoughtful response:
        """)
        return prompt


class RaR(PromptTechnique):
    """
    Rephrase and Respond (RaR) prompting technique.

    This technique asks the model to first rephrase the input in its own words
    to ensure understanding, then provide a response based on the rephrased version.
    """

    def __init__(self):
        """Initialize RaR technique."""
        super().__init__(
            name="Rephrase and Respond (RaR)",
            identifier="2.2.2.6",
            description="First rephrases the input to ensure understanding, then responds.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a RaR prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - rephrase_focus (str): Optional focus for rephrasing ("key_points", "implications", "requirements")
                - clarify_ambiguity (bool): Optional flag to address ambiguities

        Returns:
            str: Generated RaR prompt
        """
        rephrase_focus = kwargs.get("rephrase_focus", "key_points")
        clarify_ambiguity = kwargs.get("clarify_ambiguity", True)

        focus_guidance = {
            "key_points": "the main points and essential elements",
            "implications": "the underlying implications and what's being asked",
            "requirements": "the specific requirements and expected outcomes",
        }.get(rephrase_focus, "the main points and essential elements")

        clarify_str = (
            "\n- Address any ambiguities or unclear aspects"
            if clarify_ambiguity
            else ""
        )

        prompt = dedent_prompt(f"""
        I want you to first rephrase the following input in your own words to demonstrate your understanding, then provide your response.
        
        Original Input: {input_text}
        
        Step 1 - Rephrase: 
        Please rephrase this input, focusing on {focus_guidance}:{clarify_str}
        
        Step 2 - Respond:
        Based on your rephrased understanding, provide a comprehensive response:
        """)
        return prompt


class RF2(PromptTechnique):
    """
    Reason First, Format Second (RF2) prompting technique.

    This technique separates the reasoning process from the formatting,
    encouraging the model to first work through the logic and then present it clearly.
    """

    def __init__(self):
        """Initialize RF2 technique."""
        super().__init__(
            name="Reason First, Format Second (RF2)",
            identifier="2.2.2.7",
            description="Separates reasoning from formatting for clearer thought processes.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate an RF2 prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - reasoning_style (str): Optional reasoning approach
                - output_format (str): Optional desired output format
                - show_reasoning (bool): Optional flag to show or hide reasoning steps

        Returns:
            str: Generated RF2 prompt
        """
        reasoning_style = kwargs.get("reasoning_style", "step-by-step")
        output_format = kwargs.get("output_format", "clear and organized")
        show_reasoning = kwargs.get("show_reasoning", True)

        reasoning_guidance = {
            "step-by-step": "work through this step-by-step",
            "analytical": "break down and analyze the components",
            "comparative": "compare different aspects and alternatives",
            "systematic": "approach this systematically and methodically",
        }.get(reasoning_style, "work through this step-by-step")

        show_reasoning_str = (
            """
        
        Step 1 - Reasoning Phase:
        First, work through your thinking process. Don't worry about formatting yet - just focus on the logic and reasoning.
        
        Step 2 - Formatting Phase:
        Now, present your reasoning and conclusions in a clear, well-organized format.
        """
            if show_reasoning
            else """
        
        First, work through your reasoning internally, then present your final response in a clear, well-organized format.
        """
        )

        prompt = dedent_prompt(f"""
        Task: {input_text}
        
        I want you to {reasoning_guidance}, but separate your reasoning process from your final formatting.{show_reasoning_str}
        
        Target format for final response: {output_format}
        
        Begin your reasoning process:
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
            identifier="2.2.2.8",  # Unique identifier within zero-shot techniques
            description="Prompts the model to ask and answer its own questions.",
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
            "deep": "delve into underlying principles, complex interconnections, and explore nuanced aspects",
        }.get(
            depth,
            "explore key factors, important connections, and significant implications",
        )

        questions = "\n\n".join(
            [
                f"{i + 1}. [Ask a specific, focused question that helps address an important aspect of the main question{domain_str}]\n[Provide a clear, evidence-based answer to this question]"
                for i in range(num_questions)
            ]
        )

        prompt = dedent_prompt(f"""
        Main Question: {input_text}
        
        To thoroughly answer this question, I'll use a self-questioning approach. I'll identify and answer {num_questions} key follow-up questions that will help me build toward a comprehensive response. For each question, I'll {depth_guidance}.

        {questions}
        
        Now, synthesizing all the information from my self-questioning process:
        
        Final comprehensive answer to the original question:
        """)
        return prompt
