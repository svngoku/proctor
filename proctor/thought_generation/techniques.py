from typing import List, Dict, Optional
from ..base import PromptTechnique
from ..utils import dedent_prompt


class ChainOfThought(PromptTechnique):
    """
    Chain-of-Thought (CoT) encourages step-by-step reasoning for problem-solving.
    """

    def __init__(self, num_steps: int = 3):
        """
        Initialize Chain-of-Thought technique.

        Args:
            num_steps (int): Number of reasoning steps in the prompt. Defaults to 3.
        """
        super().__init__(
            name="Chain-of-Thought",
            identifier="2.2.3",
            description="Encourages step-by-step reasoning before providing an answer.",
        )
        self.num_steps = max(1, num_steps)  # Ensure at least one step

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Chain-of-Thought prompt with step-by-step reasoning guidance.

        Args:
            input_text (str): The problem or question to be solved.
            **kwargs: Additional arguments:
                - custom_instructions (str): Custom instruction text
                - approach (str): Optional reasoning approach (e.g., "analytical", "mathematical")
                - detail_level (str): Optional level of detail ("brief", "standard", "detailed")
                - include_alternatives (bool): Whether to explore alternative approaches

        Returns:
            str: Formatted Chain-of-Thought prompt.

        Raises:
            ValueError: If input_text is empty or invalid.
        """
        if not input_text or not isinstance(input_text, str):
            raise ValueError("input_text must be a non-empty string")

        approach = kwargs.get("approach", "")
        detail_level = kwargs.get("detail_level", "standard")
        include_alternatives = kwargs.get("include_alternatives", False)

        # Customize guidance based on parameters
        approach_text = f" using a {approach} approach" if approach else ""

        detail_guidance = {
            "brief": "Focus on key insights with minimal explanation.",
            "standard": "Provide balanced reasoning with moderate detail.",
            "detailed": "Explore nuances and provide comprehensive explanation.",
        }.get(detail_level, "Provide balanced reasoning with moderate detail.")

        alternatives_text = (
            "\n\nConsider at least one alternative approach or perspective before reaching your final conclusion."
            if include_alternatives
            else ""
        )

        custom_instructions = kwargs.get(
            "custom_instructions",
            f"Let's work through this{approach_text} step-by-step. {detail_guidance}{alternatives_text}",
        )

        # Create a structured prompt with clearer guidance for each step
        steps_text = ""
        for i in range(self.num_steps):
            step_num = i + 1
            if step_num == 1:
                steps_text += (
                    f"{step_num}. [Identify the key components of the problem]\n\n"
                )
            elif step_num == self.num_steps:
                steps_text += (
                    f"{step_num}. [Derive the final result based on previous steps]\n\n"
                )
            else:
                steps_text += f"{step_num}. [Apply logical reasoning to continue from previous steps]\n\n"

        prompt = dedent_prompt(f"""
        Problem/Question: {input_text}

        {custom_instructions}
        
        {steps_text}Therefore, the final answer is:
        """)
        return prompt


class ZeroShotCoT(PromptTechnique):
    """
    Zero-Shot Chain-of-Thought adds a reasoning prompt without examples.
    """

    def __init__(self):
        """Initialize Zero-Shot Chain-of-Thought technique."""
        super().__init__(
            name="Zero-Shot CoT",
            identifier="2.2.3.1",
            description="Encourages step-by-step reasoning with a simple prompt.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Zero-Shot Chain-of-Thought prompt.

        Args:
            input_text (str): The problem or question to be solved.
            **kwargs: Additional arguments:
                - custom_instructions (str): Custom instruction text
                - domain (str): Optional domain for contextualizing the reasoning
                - reasoning_style (str): Optional reasoning style ("analytical", "creative", etc.)
                - complexity (str): Optional complexity level ("simple", "intermediate", "advanced")

        Returns:
            str: Formatted Zero-Shot CoT prompt.

        Raises:
            ValueError: If input_text is empty or invalid.
        """
        if not input_text or not isinstance(input_text, str):
            raise ValueError("input_text must be a non-empty string")

        domain = kwargs.get("domain", "")
        reasoning_style = kwargs.get("reasoning_style", "")
        complexity = kwargs.get("complexity", "")

        domain_context = f" in the domain of {domain}" if domain else ""
        style_context = f" using {reasoning_style} reasoning" if reasoning_style else ""
        complexity_guidance = {
            "simple": "Break this down into basic steps, focusing on the core concepts.",
            "intermediate": "Analyze this methodically, considering important factors and relationships.",
            "advanced": "Examine this comprehensively, addressing nuances and exploring deeper implications.",
        }.get(complexity, "")

        complexity_text = f" {complexity_guidance}" if complexity else ""

        custom_instruction_base = f"Let's think step by step{domain_context}{style_context} to solve this problem:{complexity_text}"
        custom_instructions = kwargs.get("custom_instructions", custom_instruction_base)

        prompt = dedent_prompt(f"""
        Problem/Question: {input_text}

        {custom_instructions}

        1. [First, I'll identify what the problem is asking and key information provided]
        
        2. [Next, I'll determine an approach to solve this systematically]
        
        3. [I'll work through each logical step of my solution]
        
        4. [Finally, I'll verify my solution and formulate my answer]
        """)
        return prompt


class FewShotCoT(PromptTechnique):
    """
    Few-Shot Chain-of-Thought provides examples of step-by-step reasoning.
    """

    def __init__(self):
        """Initialize Few-Shot Chain-of-Thought technique."""
        super().__init__(
            name="Few-Shot CoT",
            identifier="2.2.3.2",
            description="Provides examples of step-by-step reasoning to guide problem-solving.",
        )

    def _validate_examples(self, examples: List[Dict[str, str]]) -> None:
        """
        Validate the structure and content of provided examples.

        Args:
            examples (List[Dict[str, str]]): List of example dictionaries.

        Raises:
            ValueError: If examples are invalid or missing required keys.
        """
        required_keys = {"problem", "reasoning", "answer"}
        for example in examples:
            if not isinstance(example, dict):
                raise ValueError("Each example must be a dictionary")
            if not all(key in example for key in required_keys):
                raise ValueError(f"Each example must contain {required_keys}")
            if not all(
                isinstance(example[key], str) and example[key] for key in required_keys
            ):
                raise ValueError("Example fields must be non-empty strings")

    def generate_prompt(
        self, input_text: str, examples: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> str:
        """
        Generate a Few-Shot Chain-of-Thought prompt with examples of reasoning.

        Args:
            input_text (str): The problem or question to be solved.
            examples (Optional[List[Dict[str, str]]]): Examples with 'problem', 'reasoning', and 'answer' keys.
            **kwargs: Additional arguments (e.g., custom instructions).

        Returns:
            str: Formatted Few-Shot CoT prompt.

        Raises:
            ValueError: If input_text is empty or examples are invalid.
        """
        if not input_text or not isinstance(input_text, str):
            raise ValueError("input_text must be a non-empty string")

        # Use default examples if none provided
        if examples is None:
            examples = [
                {
                    "problem": "If John has 5 apples and gives 2 to Mary, how many does he have left?",
                    "reasoning": "John starts with 5 apples. He gives 2 apples to Mary. Subtracting the apples given away, 5 - 2 = 3 apples remain.",
                    "answer": "3 apples",
                },
                {
                    "problem": "If a train travels 120 km in 2 hours, what is its speed?",
                    "reasoning": "To find speed, use the formula: speed = distance ÷ time. The train travels 120 km in 2 hours. Therefore, speed = 120 km ÷ 2 hours = 60 km/hour.",
                    "answer": "60 km/hour",
                },
            ]

        # Validate examples
        self._validate_examples(examples)

        # Format examples into prompt
        examples_text = "\n\n".join(
            [
                f"Problem: {example['problem']}\n\nReasoning: {example['reasoning']}\n\nAnswer: {example['answer']}"
                for example in examples
            ]
        )

        domain = kwargs.get("domain", "")
        focus_areas = kwargs.get("focus_areas", [])

        domain_text = f" in {domain}" if domain else ""
        focus_text = ""
        if focus_areas:
            focus_text = "\n- Pay special attention to: " + ", ".join(focus_areas)

        custom_instructions = kwargs.get(
            "custom_instructions",
            f"Use the same step-by-step reasoning approach as shown in the examples to solve the following problem{domain_text}:",
        )

        prompt = dedent_prompt(f"""
        Below are examples of problems solved using effective step-by-step reasoning. Study these patterns carefully:

        {examples_text}

        {custom_instructions}
        {focus_text}

        Problem: {input_text}

        I'll solve this by following a similar reasoning process:
        1. First, I'll understand what the problem is asking
        2. Then, I'll identify the key information and constraints
        3. Next, I'll apply a systematic approach similar to the examples
        4. Finally, I'll derive the answer through careful reasoning

        Reasoning:
        """)
        return prompt


class AnalogicalPrompting(PromptTechnique):
    """
    Analogical Prompting uses analogies to guide reasoning and problem-solving.

    This technique leverages analogical reasoning by drawing parallels between
    the current problem and similar, well-understood situations or examples.
    """

    def __init__(self):
        """Initialize Analogical Prompting technique."""
        super().__init__(
            name="Analogical Prompting",
            identifier="2.2.3.1",
            description="Uses analogies to guide reasoning and problem-solving.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate an analogical prompting prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - analogy_domain (str): Domain to draw analogies from
                - num_analogies (int): Number of analogies to use
                - analogy_examples (List[str]): Specific analogies to use

        Returns:
            str: Generated analogical prompt
        """
        analogy_domain = kwargs.get("analogy_domain", "everyday life")
        num_analogies = kwargs.get("num_analogies", 2)
        analogy_examples = kwargs.get("analogy_examples", [])

        if not analogy_examples:
            analogy_examples = [
                f"Think of this like {analogy_domain} - what similar situations have you encountered?",
                f"Consider an analogy from {analogy_domain} that shares key characteristics with this problem.",
            ][:num_analogies]

        analogies_text = "\n".join([f"- {analogy}" for analogy in analogy_examples])

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        To solve this problem, I'll use analogical reasoning by drawing parallels to familiar situations:
        
        {analogies_text}
        
        Now, let me work through this step by step using these analogies:
        
        1. First, I'll identify the key elements of the current problem
        2. Then, I'll map these elements to my chosen analogy
        3. Next, I'll apply the reasoning patterns from the analogous situation
        4. Finally, I'll translate the insights back to solve the original problem
        
        Analogical reasoning:
        """)
        return prompt


class StepBackPrompting(PromptTechnique):
    """
    Step-Back Prompting encourages taking a step back to consider higher-level principles.

    This technique prompts the model to first consider broader concepts and principles
    before diving into specific problem-solving steps.
    """

    def __init__(self):
        """Initialize Step-Back Prompting technique."""
        super().__init__(
            name="Step-Back Prompting",
            identifier="2.2.3.1",
            description="Encourages considering higher-level principles before detailed problem-solving.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a step-back prompting prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - abstraction_level (str): Level of abstraction ("principles", "concepts", "patterns")
                - domain_knowledge (str): Relevant domain knowledge to consider

        Returns:
            str: Generated step-back prompt
        """
        abstraction_level = kwargs.get("abstraction_level", "principles")
        domain_knowledge = kwargs.get("domain_knowledge", "")

        domain_text = f" in {domain_knowledge}" if domain_knowledge else ""

        abstraction_guidance = {
            "principles": "fundamental principles and underlying laws",
            "concepts": "key concepts and theoretical frameworks",
            "patterns": "common patterns and recurring themes",
        }.get(abstraction_level, "fundamental principles")

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        Before diving into the specifics, let me step back and consider the bigger picture.
        
        Step Back - Higher Level Analysis:
        1. What are the {abstraction_guidance}{domain_text} that apply to this situation?
        2. What general category or type of problem is this?
        3. What broader context or framework should I consider?
        4. Are there any overarching patterns or rules that govern this domain?
        
        Now, with this higher-level understanding, let me approach the specific problem:
        
        Detailed Problem-Solving:
        1. How do the higher-level principles apply to this specific case?
        2. What specific steps follow from the general framework?
        3. How can I systematically work through the details?
        4. What is my final answer based on this principled approach?
        """)
        return prompt


class ThreadOfThought(PromptTechnique):
    """
    Thread-of-Thought (ThoT) maintains coherent reasoning threads across complex problems.

    This technique helps maintain logical consistency and coherence when dealing
    with multi-faceted problems that require tracking multiple reasoning threads.
    """

    def __init__(self):
        """Initialize Thread-of-Thought technique."""
        super().__init__(
            name="Thread-of-Thought (ThoT)",
            identifier="2.2.3.1",
            description="Maintains coherent reasoning threads across complex problems.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a thread-of-thought prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - num_threads (int): Number of reasoning threads to maintain
                - thread_topics (List[str]): Specific topics for each thread

        Returns:
            str: Generated thread-of-thought prompt
        """
        num_threads = kwargs.get("num_threads", 3)
        thread_topics = kwargs.get("thread_topics", [])

        if not thread_topics:
            thread_topics = [f"Thread {i + 1}" for i in range(num_threads)]

        threads_setup = "\n".join(
            [
                f"Thread {i + 1} ({topic}): [Focus on {topic.lower()} aspects]"
                for i, topic in enumerate(thread_topics[:num_threads])
            ]
        )

        threads_development = "\n\n".join(
            [
                f"Thread {i + 1} Development:\n[Develop reasoning for {topic.lower()}]"
                for i, topic in enumerate(thread_topics[:num_threads])
            ]
        )

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll approach this complex problem by maintaining multiple coherent reasoning threads:
        
        Thread Setup:
        {threads_setup}
        
        {threads_development}
        
        Thread Integration:
        Now I'll weave these threads together to form a comprehensive solution:
        1. How do the different threads connect and support each other?
        2. Are there any contradictions between threads that need resolution?
        3. What insights emerge from considering all threads together?
        4. What is the integrated final answer?
        """)
        return prompt


class TabCoT(PromptTechnique):
    """
    Tab-CoT organizes reasoning into tabular format for systematic analysis.

    This technique structures the reasoning process using tables or organized
    formats to ensure systematic coverage of all relevant aspects.
    """

    def __init__(self):
        """Initialize Tab-CoT technique."""
        super().__init__(
            name="Tab-CoT",
            identifier="2.2.3.1",
            description="Organizes reasoning into tabular format for systematic analysis.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Tab-CoT prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - table_headers (List[str]): Headers for the reasoning table
                - analysis_dimensions (List[str]): Dimensions to analyze

        Returns:
            str: Generated Tab-CoT prompt
        """
        table_headers = kwargs.get(
            "table_headers", ["Aspect", "Analysis", "Conclusion"]
        )
        analysis_dimensions = kwargs.get(
            "analysis_dimensions",
            ["Key Facts", "Assumptions", "Implications", "Solution"],
        )

        headers_text = " | ".join(table_headers)
        separator = " | ".join(["---"] * len(table_headers))

        table_rows = []
        for dimension in analysis_dimensions:
            row_cells = [dimension] + ["[To be filled]"] * (len(table_headers) - 1)
            table_rows.append(" | ".join(row_cells))

        table_template = f"{headers_text}\n{separator}\n" + "\n".join(table_rows)

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll organize my reasoning systematically using a tabular approach:
        
        Reasoning Table:
        {table_template}
        
        Now let me fill in this table systematically:
        
        Step 1: Analyze each dimension thoroughly
        Step 2: Fill in the reasoning table with detailed analysis
        Step 3: Draw connections between different rows
        Step 4: Synthesize findings into a final answer
        
        Completed Analysis:
        """)
        return prompt


class ActivePrompt(PromptTechnique):
    """
    Active-Prompt adapts the prompting strategy based on problem characteristics.

    This technique dynamically adjusts the prompting approach based on the
    specific characteristics and requirements of the problem at hand.
    """

    def __init__(self):
        """Initialize Active-Prompt technique."""
        super().__init__(
            name="Active-Prompt",
            identifier="2.2.3.2",
            description="Adapts prompting strategy based on problem characteristics.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate an active prompt that adapts to problem characteristics.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - problem_type (str): Type of problem detected
                - complexity_level (str): Complexity level assessment
                - required_skills (List[str]): Skills required for the problem

        Returns:
            str: Generated adaptive prompt
        """
        problem_type = kwargs.get("problem_type", "general")
        complexity_level = kwargs.get("complexity_level", "medium")
        required_skills = kwargs.get("required_skills", ["analysis", "reasoning"])

        # Adapt strategy based on problem characteristics
        if problem_type == "mathematical":
            strategy = "systematic calculation and formula application"
        elif problem_type == "logical":
            strategy = "step-by-step logical deduction"
        elif problem_type == "creative":
            strategy = "divergent thinking and idea generation"
        elif problem_type == "analytical":
            strategy = "structured analysis and evidence evaluation"
        else:
            strategy = "comprehensive problem-solving approach"

        complexity_guidance = {
            "low": "Focus on clear, direct reasoning steps",
            "medium": "Balance thoroughness with efficiency",
            "high": "Use comprehensive analysis with multiple verification steps",
        }.get(complexity_level, "Balance thoroughness with efficiency")

        skills_text = ", ".join(required_skills)

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        Problem Analysis:
        - Type: {problem_type}
        - Complexity: {complexity_level}
        - Required skills: {skills_text}
        
        Adaptive Strategy: {strategy}
        Approach: {complexity_guidance}
        
        Tailored Solution Process:
        1. [Apply problem-type specific analysis methods]
        2. [Use complexity-appropriate reasoning depth]
        3. [Leverage the required skills: {skills_text}]
        4. [Verify solution using type-specific validation methods]
        
        Solution:
        """)
        return prompt


class AutoCoT(PromptTechnique):
    """
    Auto-CoT automatically generates chain-of-thought reasoning.

    This technique automatically constructs reasoning chains without requiring
    manual specification of reasoning steps or examples.
    """

    def __init__(self):
        """Initialize Auto-CoT technique."""
        super().__init__(
            name="Auto-CoT",
            identifier="2.2.3.2",
            description="Automatically generates chain-of-thought reasoning.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate an auto-CoT prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - reasoning_depth (str): Depth of automatic reasoning
                - auto_verification (bool): Whether to include automatic verification

        Returns:
            str: Generated auto-CoT prompt
        """
        reasoning_depth = kwargs.get("reasoning_depth", "standard")
        auto_verification = kwargs.get("auto_verification", True)

        depth_guidance = {
            "shallow": "Generate 2-3 key reasoning steps",
            "standard": "Generate 4-5 comprehensive reasoning steps",
            "deep": "Generate 6+ detailed reasoning steps with sub-analysis",
        }.get(reasoning_depth, "Generate 4-5 comprehensive reasoning steps")

        verification_text = (
            """
        
        Auto-Verification:
        - Check reasoning consistency
        - Validate logical flow
        - Confirm answer accuracy
        """
            if auto_verification
            else ""
        )

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll automatically generate a comprehensive reasoning chain for this problem.
        
        Auto-Generated Reasoning Chain:
        {depth_guidance}
        
        [The model will automatically identify the optimal reasoning structure and generate appropriate steps]
        
        1. [Auto-identify problem type and requirements]
        2. [Auto-generate relevant reasoning steps]
        3. [Auto-apply appropriate problem-solving methods]
        4. [Auto-synthesize solution from reasoning chain]
        {verification_text}
        
        Automated Solution:
        """)
        return prompt


class ComplexityBased(PromptTechnique):
    """
    Complexity-Based prompting adjusts reasoning depth based on problem complexity.

    This technique scales the reasoning approach based on the assessed complexity
    of the problem, using simpler approaches for simple problems and more
    sophisticated reasoning for complex ones.
    """

    def __init__(self):
        """Initialize Complexity-Based technique."""
        super().__init__(
            name="Complexity-Based",
            identifier="2.2.3.2",
            description="Adjusts reasoning depth based on problem complexity.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a complexity-based prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - complexity_score (float): Complexity score (0-1)
                - auto_assess (bool): Whether to auto-assess complexity

        Returns:
            str: Generated complexity-based prompt
        """
        complexity_score = kwargs.get("complexity_score", 0.5)
        auto_assess = kwargs.get("auto_assess", True)

        if auto_assess:
            assessment_text = """
        First, let me assess the complexity of this problem:
        - Number of variables and constraints
        - Required domain knowledge
        - Interdependencies between components
        - Potential solution approaches
        
        Based on this assessment, I'll choose the appropriate reasoning depth.
        """
        else:
            assessment_text = f"Given complexity score: {complexity_score}"

        if complexity_score < 0.3:
            approach = "Simple, direct reasoning with minimal steps"
        elif complexity_score < 0.7:
            approach = "Moderate reasoning with systematic analysis"
        else:
            approach = "Deep, multi-layered reasoning with comprehensive analysis"

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        Complexity Assessment:
        {assessment_text}
        
        Chosen Approach: {approach}
        
        Complexity-Adapted Solution:
        [The reasoning depth and structure will be automatically adjusted based on the complexity assessment]
        
        Solution:
        """)
        return prompt


class Contrastive(PromptTechnique):
    """
    Contrastive prompting uses contrasting examples or approaches for better understanding.

    This technique leverages contrasts and comparisons to highlight key differences
    and improve reasoning through comparative analysis.
    """

    def __init__(self):
        """Initialize Contrastive technique."""
        super().__init__(
            name="Contrastive",
            identifier="2.2.3.2",
            description="Uses contrasting examples or approaches for better understanding.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a contrastive prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - contrast_examples (List[str]): Specific contrasting examples
                - contrast_dimensions (List[str]): Dimensions to contrast

        Returns:
            str: Generated contrastive prompt
        """
        contrast_examples = kwargs.get("contrast_examples", [])
        contrast_dimensions = kwargs.get(
            "contrast_dimensions", ["approach", "assumptions", "outcomes"]
        )

        if not contrast_examples:
            contrast_examples = [
                "What would happen if we approached this differently?",
                "How does this compare to the opposite scenario?",
                "What are the key differences between alternative solutions?",
            ]

        dimensions_text = ", ".join(contrast_dimensions)
        contrasts_text = "\n".join([f"- {example}" for example in contrast_examples])

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll use contrastive reasoning to better understand this problem by examining different perspectives and approaches.
        
        Contrastive Analysis:
        {contrasts_text}
        
        Systematic Comparison:
        I'll analyze contrasts across these dimensions: {dimensions_text}
        
        1. Primary Approach vs. Alternative Approaches
        2. Key Assumptions vs. Different Assumptions  
        3. Expected Outcomes vs. Alternative Outcomes
        4. Advantages vs. Disadvantages
        
        Synthesis:
        Based on this contrastive analysis, what insights emerge?
        What is the most robust solution considering all contrasts?
        
        Final Answer:
        """)
        return prompt


class MemoryOfThought(PromptTechnique):
    """
    Memory-of-Thought maintains and references previous reasoning steps.

    This technique explicitly tracks and references previous reasoning steps
    to maintain consistency and build upon earlier insights.
    """

    def __init__(self):
        """Initialize Memory-of-Thought technique."""
        super().__init__(
            name="Memory-of-Thought",
            identifier="2.2.3.2",
            description="Maintains and references previous reasoning steps.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a memory-of-thought prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - memory_capacity (int): Number of previous steps to remember
                - reference_style (str): How to reference previous steps

        Returns:
            str: Generated memory-of-thought prompt
        """
        memory_capacity = kwargs.get("memory_capacity", 5)
        reference_style = kwargs.get("reference_style", "explicit")

        reference_instruction = {
            "explicit": "Explicitly reference and build upon previous steps",
            "implicit": "Implicitly maintain consistency with previous reasoning",
            "summary": "Periodically summarize and reference key previous insights",
        }.get(reference_style, "Explicitly reference and build upon previous steps")

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll solve this using Memory-of-Thought, maintaining awareness of previous reasoning steps.
        
        Memory Management: {reference_instruction}
        Memory Capacity: Track last {memory_capacity} reasoning steps
        
        Step 1: [Initial analysis - MEMORY: Store key insights]
        
        Step 2: [Building on Step 1 - MEMORY: Reference previous insights]
        
        Step 3: [Further development - MEMORY: Integrate with Steps 1-2]
        
        Step 4: [Advanced reasoning - MEMORY: Synthesize all previous steps]
        
        Step 5: [Final solution - MEMORY: Validate against all previous reasoning]
        
        Memory Summary:
        [Key insights from all steps that inform the final answer]
        
        Final Answer:
        """)
        return prompt


class UncertaintyRouted(PromptTechnique):
    """
    Uncertainty-Routed CoT routes reasoning based on uncertainty levels.

    This technique adapts the reasoning approach based on the level of uncertainty
    in different aspects of the problem, using more thorough analysis for uncertain areas.
    """

    def __init__(self):
        """Initialize Uncertainty-Routed technique."""
        super().__init__(
            name="Uncertainty-Routed CoT",
            identifier="2.2.3.2",
            description="Routes reasoning based on uncertainty levels.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate an uncertainty-routed prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - uncertainty_threshold (float): Threshold for high uncertainty
                - routing_strategy (str): Strategy for handling uncertainty

        Returns:
            str: Generated uncertainty-routed prompt
        """
        uncertainty_threshold = kwargs.get("uncertainty_threshold", 0.7)
        routing_strategy = kwargs.get("routing_strategy", "adaptive")

        strategy_guidance = {
            "adaptive": "Dynamically adjust reasoning depth based on uncertainty",
            "conservative": "Use thorough analysis for all uncertain aspects",
            "efficient": "Focus detailed analysis only on highest uncertainty areas",
        }.get(
            routing_strategy, "Dynamically adjust reasoning depth based on uncertainty"
        )

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll use Uncertainty-Routed reasoning to adapt my approach based on confidence levels.
        
        Uncertainty Assessment:
        1. Identify aspects with high uncertainty (>{uncertainty_threshold})
        2. Identify aspects with low uncertainty (<{uncertainty_threshold})
        3. Route reasoning depth accordingly
        
        Routing Strategy: {strategy_guidance}
        
        High Uncertainty Areas:
        [Detailed analysis with multiple verification steps]
        
        Medium Uncertainty Areas:
        [Standard reasoning with validation checks]
        
        Low Uncertainty Areas:
        [Efficient reasoning with basic verification]
        
        Uncertainty-Aware Solution:
        [Final answer with confidence levels for different components]
        """)
        return prompt
