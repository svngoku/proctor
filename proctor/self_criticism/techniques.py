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
            description="Reviews and verifies each step of reasoning.",
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
                "appropriateness of methods used",
            ]

        verification_level = {
            "basic": "Check for obvious errors and inconsistencies.",
            "thorough": "Carefully examine assumptions, methods, and conclusions for validity and completeness.",
            "rigorous": "Conduct an exhaustive verification, challenging every aspect of the solution with alternative perspectives.",
        }.get(
            verification_intensity,
            "Carefully examine assumptions, methods, and conclusions for validity and completeness.",
        )

        counterexample_text = (
            "\n- Actively attempt to construct counterexamples or cases where this solution might fail"
            if include_counterexamples
            else ""
        )

        # Generate solution steps
        solution_steps = "\n".join(
            [f"{i + 1}. [Solution step {i + 1}]" for i in range(num_steps)]
        )

        # Generate verification sections
        verification_sections = ""
        for i in range(num_steps):
            step_num = i + 1
            previous_steps_ref = (
                f" based on step{'s' if i > 0 else ''} {', '.join(str(j + 1) for j in range(i))}"
                if i > 0
                else ""
            )

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
        I will first generate an initial solution, then critically verify each step using a {verification_intensity} approach. I will check specifically for issues with: {", ".join(verification_aspects)}.
        
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


class SelfCalibration(PromptTechnique):
    """
    Self-Calibration adjusts confidence levels based on performance assessment.

    This technique helps the model calibrate its confidence levels by assessing
    its own performance and adjusting confidence accordingly.
    """

    def __init__(self):
        """Initialize Self-Calibration technique."""
        super().__init__(
            name="Self-Calibration",
            identifier="2.2.6",
            description="Adjusts confidence levels based on performance assessment.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Self-Calibration prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - calibration_factors (List[str]): Factors to consider for calibration
                - confidence_scale (str): Scale for confidence assessment

        Returns:
            str: Generated Self-Calibration prompt
        """
        calibration_factors = kwargs.get(
            "calibration_factors",
            ["complexity", "familiarity", "evidence_quality", "reasoning_depth"],
        )
        confidence_scale = kwargs.get("confidence_scale", "0-100")

        factors_text = "\n".join(
            [
                f"- {factor.title()}: [Assess {factor} level]"
                for factor in calibration_factors
            ]
        )

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll solve this problem while continuously calibrating my confidence based on multiple factors.
        
        Initial Solution:
        [Provide initial solution to the problem]
        
        Self-Calibration Assessment:
        {factors_text}
        
        Confidence Calibration Process:
        1. Initial Confidence: [Rate on {confidence_scale} scale]
        2. Factor Analysis: [Adjust based on calibration factors]
        3. Historical Performance: [Consider past performance on similar problems]
        4. Uncertainty Quantification: [Identify sources of uncertainty]
        
        Calibrated Confidence Level: [Final confidence rating with justification]
        
        Calibrated Final Answer:
        [Final answer with appropriately calibrated confidence]
        """)
        return prompt


class SelfRefine(PromptTechnique):
    """
    Self-Refine iteratively improves solutions through self-criticism.

    This technique uses iterative self-criticism and refinement to progressively
    improve the quality of solutions.
    """

    def __init__(self):
        """Initialize Self-Refine technique."""
        super().__init__(
            name="Self-Refine",
            identifier="2.2.6",
            description="Iteratively improves solutions through self-criticism.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Self-Refine prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - refinement_iterations (int): Number of refinement iterations
                - refinement_criteria (List[str]): Criteria for refinement

        Returns:
            str: Generated Self-Refine prompt
        """
        refinement_iterations = kwargs.get("refinement_iterations", 3)
        refinement_criteria = kwargs.get(
            "refinement_criteria", ["accuracy", "completeness", "clarity", "efficiency"]
        )

        criteria_text = ", ".join(refinement_criteria)

        iterations_text = []
        for i in range(refinement_iterations):
            iterations_text.append(f"""
        Iteration {i + 1}:
        Current Solution: [Present current solution]
        Self-Criticism: [Identify weaknesses and areas for improvement]
        Refinement: [Improve solution based on criticism]
        Quality Assessment: [Evaluate improvement using criteria: {criteria_text}]
        """)

        iterations_content = "\n".join(iterations_text)

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll use Self-Refine to iteratively improve my solution through self-criticism.
        
        Refinement Criteria: {criteria_text}
        Number of Iterations: {refinement_iterations}
        
        Initial Solution:
        [Provide initial solution attempt]
        
        Refinement Process:
        {iterations_content}
        
        Final Refined Solution:
        [Present the final, refined solution after all iterations]
        
        Refinement Summary:
        [Summarize key improvements made through the refinement process]
        """)
        return prompt


class SelfVerification(PromptTechnique):
    """
    Self-Verification systematically verifies solution correctness.

    This technique implements systematic self-verification procedures to
    check the correctness and validity of solutions.
    """

    def __init__(self):
        """Initialize Self-Verification technique."""
        super().__init__(
            name="Self-Verification",
            identifier="2.2.6",
            description="Systematically verifies solution correctness.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Self-Verification prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - verification_methods (List[str]): Specific verification methods to use
                - verification_depth (str): Depth of verification

        Returns:
            str: Generated Self-Verification prompt
        """
        verification_methods = kwargs.get(
            "verification_methods",
            ["logical_check", "consistency_check", "boundary_check", "sanity_check"],
        )
        verification_depth = kwargs.get("verification_depth", "thorough")

        depth_guidance = {
            "basic": "Perform essential verification checks",
            "thorough": "Conduct comprehensive verification across multiple dimensions",
            "exhaustive": "Perform extensive verification with multiple alternative approaches",
        }.get(verification_depth, "Conduct comprehensive verification")

        methods_text = "\n".join(
            [
                f"- {method.replace('_', ' ').title()}: [Apply {method.replace('_', ' ')} verification]"
                for method in verification_methods
            ]
        )

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll solve this problem and then systematically verify the solution.
        
        Verification Depth: {verification_depth} - {depth_guidance}
        
        Initial Solution:
        [Provide initial solution]
        
        Systematic Verification:
        {methods_text}
        
        Cross-Verification:
        1. Alternative Solution Path: [Solve using different approach]
        2. Result Comparison: [Compare results from different approaches]
        3. Discrepancy Analysis: [Analyze any differences found]
        4. Resolution: [Resolve discrepancies and determine correct solution]
        
        Verification Conclusion:
        [Final verified solution with confidence assessment]
        """)
        return prompt


class ReverseCoT(PromptTechnique):
    """
    Reverse-CoT works backwards from conclusions to verify reasoning.

    This technique starts from the conclusion and works backwards through
    the reasoning chain to verify logical consistency and identify potential errors.
    """

    def __init__(self):
        """Initialize Reverse-CoT technique."""
        super().__init__(
            name="Reverse-CoT",
            identifier="2.2.6",
            description="Works backwards from conclusions to verify reasoning.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Reverse-CoT prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - reverse_steps (int): Number of reverse reasoning steps
                - verification_focus (str): Focus area for reverse verification

        Returns:
            str: Generated Reverse-CoT prompt
        """
        reverse_steps = kwargs.get("reverse_steps", 4)
        verification_focus = kwargs.get("verification_focus", "logical_consistency")

        focus_guidance = {
            "logical_consistency": "Focus on logical flow and consistency",
            "assumption_validity": "Focus on validating underlying assumptions",
            "evidence_support": "Focus on evidence supporting each step",
            "alternative_paths": "Focus on exploring alternative reasoning paths",
        }.get(verification_focus, "Focus on overall reasoning quality")

        reverse_steps_text = []
        for i in range(reverse_steps):
            step_num = reverse_steps - i
            reverse_steps_text.append(f"""
        Reverse Step {i + 1} (Original Step {step_num}):
        [Work backwards from the conclusion to verify step {step_num}]
        [Check: Does this step logically lead to the next step?]
        [Verify: Are the assumptions and reasoning valid?]
        """)

        reverse_content = "\n".join(reverse_steps_text)

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll solve this problem using forward reasoning, then verify using Reverse-CoT.
        
        Forward Solution:
        [Provide complete forward reasoning solution]
        
        Reverse Verification Process:
        Verification Focus: {verification_focus} - {focus_guidance}
        
        Starting from the conclusion, I'll work backwards through each step:
        {reverse_content}
        
        Reverse Verification Analysis:
        1. Consistency Check: [Are all reverse steps consistent with forward reasoning?]
        2. Gap Identification: [Are there any logical gaps or jumps?]
        3. Assumption Validation: [Are all assumptions properly supported?]
        4. Alternative Paths: [Could different reasoning lead to the same conclusion?]
        
        Reverse-Verified Solution:
        [Final solution verified through reverse reasoning]
        """)
        return prompt


class CumulativeReasoning(PromptTechnique):
    """
    Cumulative Reasoning builds and validates reasoning incrementally.

    This technique builds reasoning step by step, validating each addition
    to ensure cumulative consistency and correctness.
    """

    def __init__(self):
        """Initialize Cumulative Reasoning technique."""
        super().__init__(
            name="Cumulative Reasoning",
            identifier="2.2.6",
            description="Builds and validates reasoning incrementally.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Cumulative Reasoning prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - cumulative_steps (int): Number of cumulative reasoning steps
                - validation_checkpoints (List[int]): Steps at which to perform validation

        Returns:
            str: Generated Cumulative Reasoning prompt
        """
        cumulative_steps = kwargs.get("cumulative_steps", 5)
        validation_checkpoints = kwargs.get(
            "validation_checkpoints", [2, 4, cumulative_steps]
        )

        steps_text = []
        for i in range(cumulative_steps):
            step_num = i + 1
            is_checkpoint = step_num in validation_checkpoints

            validation_text = (
                """
        
        VALIDATION CHECKPOINT:
        - Cumulative Consistency: [Check consistency of all steps so far]
        - Progress Assessment: [Evaluate progress toward solution]
        - Error Detection: [Identify any errors in cumulative reasoning]
        - Course Correction: [Make any necessary adjustments]
        """
                if is_checkpoint
                else ""
            )

            steps_text.append(f"""
        Step {step_num}:
        [Add new reasoning element]
        [Integrate with previous steps: {", ".join([str(j) for j in range(1, step_num)])}]
        [Validate integration and consistency]{validation_text}
        """)

        steps_content = "\n".join(steps_text)

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll use Cumulative Reasoning to build the solution incrementally with validation.
        
        Validation Checkpoints: Steps {", ".join(map(str, validation_checkpoints))}
        
        Incremental Reasoning Process:
        {steps_content}
        
        Final Cumulative Validation:
        1. Complete Integration: [Verify all steps work together seamlessly]
        2. Comprehensive Check: [Validate the entire cumulative reasoning chain]
        3. Solution Completeness: [Ensure the solution fully addresses the problem]
        4. Quality Assessment: [Evaluate overall solution quality]
        
        Cumulatively Validated Solution:
        [Final solution built through validated incremental reasoning]
        """)
        return prompt
