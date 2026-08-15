"""
Common DSPy Signatures for prompt engineering techniques.

This module provides reusable Signature classes for common patterns
in prompt engineering. These signatures define structured input/output
contracts that DSPy can use for optimization and type safety.
"""

import dspy
from typing import List, Optional


# ============================================================================
# Thought Generation Signatures
# ============================================================================

class ChainOfThoughtSignature(dspy.Signature):
    """
    Chain of Thought reasoning with step-by-step breakdown.

    This signature structures the CoT pattern into clear reasoning
    steps and a final answer.
    """

    problem: str = dspy.InputField(desc="The problem or question to solve")
    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning process, showing all intermediate steps"
    )
    answer: str = dspy.OutputField(desc="The final answer or conclusion")


class AnalogicalReasoningSignature(dspy.Signature):
    """Reasoning by analogy."""

    problem: str = dspy.InputField(desc="The problem to solve")
    analogy: str = dspy.OutputField(desc="A relevant analogy to help understand the problem")
    reasoning: str = dspy.OutputField(desc="How the analogy applies")
    answer: str = dspy.OutputField(desc="Solution based on the analogy")


class StepBackSignature(dspy.Signature):
    """Step-back prompting to abstract the problem."""

    problem: str = dspy.InputField(desc="The specific problem")
    abstract_problem: str = dspy.OutputField(desc="Higher-level abstraction of the problem")
    general_principle: str = dspy.OutputField(desc="General principles that apply")
    specific_answer: str = dspy.OutputField(desc="Answer to the specific problem")


# ============================================================================
# Role-Based Signatures
# ============================================================================

class RoleBasedSignature(dspy.Signature):
    """Response from a specific role or persona."""

    problem: str = dspy.InputField(desc="The task or question")
    role: str = dspy.InputField(desc="The role to assume (e.g., 'expert', 'teacher')")
    context: str = dspy.OutputField(desc="Context from the role's perspective")
    response: str = dspy.OutputField(desc="Response as the specified role")


class ExpertAnalysisSignature(dspy.Signature):
    """Expert-level analysis with detailed breakdown."""

    problem: str = dspy.InputField(desc="The problem requiring expert analysis")
    domain: str = dspy.InputField(desc="Domain of expertise", default="general")
    analysis: str = dspy.OutputField(desc="Detailed expert analysis")
    recommendation: str = dspy.OutputField(desc="Expert recommendation or solution")
    confidence: str = dspy.OutputField(desc="Confidence level in the analysis")


# ============================================================================
# Decomposition Signatures
# ============================================================================

class ProblemDecompositionSignature(dspy.Signature):
    """Decompose a complex problem into smaller sub-problems."""

    problem: str = dspy.InputField(desc="The complex problem to decompose")
    subproblems: str = dspy.OutputField(
        desc="List of smaller, manageable sub-problems (one per line)"
    )
    approach: str = dspy.OutputField(desc="Overall approach to solving the problem")


class LeastToMostSignature(dspy.Signature):
    """Solve problems from easiest to hardest."""

    problem: str = dspy.InputField(desc="The problem to solve")
    ordered_subproblems: str = dspy.OutputField(
        desc="Sub-problems ordered from easiest to hardest"
    )
    solution_strategy: str = dspy.OutputField(desc="Strategy for progressive solving")


class PlanAndSolveSignature(dspy.Signature):
    """First plan, then execute the plan."""

    problem: str = dspy.InputField(desc="The problem to solve")
    plan: str = dspy.OutputField(desc="Step-by-step plan to solve the problem")
    execution: str = dspy.OutputField(desc="Execution of the plan with results")
    answer: str = dspy.OutputField(desc="Final answer")


# ============================================================================
# Verification & Self-Criticism Signatures
# ============================================================================

class VerificationSignature(dspy.Signature):
    """Verify a proposed solution."""

    problem: str = dspy.InputField(desc="The original problem")
    proposed_answer: str = dspy.InputField(desc="The proposed answer to verify")
    verification: str = dspy.OutputField(desc="Verification process and checks")
    is_correct: str = dspy.OutputField(desc="Whether the answer is correct (yes/no)")
    corrected_answer: str = dspy.OutputField(
        desc="Corrected answer if original was wrong, or same answer if correct"
    )


class SelfRefineSignature(dspy.Signature):
    """Iteratively refine a response."""

    problem: str = dspy.InputField(desc="The original problem")
    current_answer: str = dspy.InputField(desc="Current answer to refine")
    critique: str = dspy.OutputField(desc="Critique of the current answer")
    refined_answer: str = dspy.OutputField(desc="Improved version of the answer")


class ChainOfVerificationSignature(dspy.Signature):
    """Multi-step verification process."""

    problem: str = dspy.InputField(desc="The problem")
    answer: str = dspy.InputField(desc="The answer to verify")
    verification_questions: str = dspy.OutputField(
        desc="Questions to ask for verification (one per line)"
    )
    verification_answers: str = dspy.OutputField(desc="Answers to verification questions")
    final_verdict: str = dspy.OutputField(desc="Final verification verdict")


# ============================================================================
# Style & Formatting Signatures
# ============================================================================

class StyledOutputSignature(dspy.Signature):
    """Generate output in a specific style."""

    content: str = dspy.InputField(desc="The content to style")
    style: str = dspy.InputField(
        desc="Desired style (e.g., 'formal', 'casual', 'technical')"
    )
    styled_output: str = dspy.OutputField(desc="Content formatted in the requested style")


class EmotionalResponseSignature(dspy.Signature):
    """Response with specific emotional tone."""

    problem: str = dspy.InputField(desc="The task or question")
    emotion: str = dspy.InputField(
        desc="Emotional tone (e.g., 'excited', 'calm', 'serious')"
    )
    response: str = dspy.OutputField(desc="Response with the specified emotional tone")


# ============================================================================
# Few-Shot Learning Signatures
# ============================================================================

class FewShotPredictionSignature(dspy.Signature):
    """Make predictions based on examples."""

    examples: str = dspy.InputField(desc="Example input-output pairs")
    query: str = dspy.InputField(desc="New query to answer")
    pattern: str = dspy.OutputField(desc="Identified pattern from examples")
    prediction: str = dspy.OutputField(desc="Prediction for the query")


class ExampleSelectionSignature(dspy.Signature):
    """Select the most relevant examples."""

    query: str = dspy.InputField(desc="Query to find examples for")
    available_examples: str = dspy.InputField(desc="Pool of available examples")
    selected_examples: str = dspy.OutputField(desc="Most relevant examples")
    selection_reasoning: str = dspy.OutputField(desc="Why these examples were selected")


# ============================================================================
# Ensembling Signatures
# ============================================================================

class ConsensusSignature(dspy.Signature):
    """Reach consensus from multiple answers."""

    problem: str = dspy.InputField(desc="The original problem")
    candidate_answers: str = dspy.InputField(desc="Multiple candidate answers")
    analysis: str = dspy.OutputField(desc="Analysis of the different answers")
    consensus_answer: str = dspy.OutputField(desc="The consensus answer")
    confidence: str = dspy.OutputField(desc="Confidence in the consensus")


class DiverseReasoningSignature(dspy.Signature):
    """Generate diverse reasoning approaches."""

    problem: str = dspy.InputField(desc="The problem to solve")
    approach: str = dspy.InputField(desc="Which approach to use (e.g., 'analytical', 'creative')")
    reasoning: str = dspy.OutputField(desc="Reasoning using the specified approach")
    answer: str = dspy.OutputField(desc="Answer from this approach")


# ============================================================================
# Specialized Signatures
# ============================================================================

class CodeGenerationSignature(dspy.Signature):
    """Generate code to solve a problem."""

    problem: str = dspy.InputField(desc="Description of what the code should do")
    language: str = dspy.InputField(desc="Programming language", default="python")
    explanation: str = dspy.OutputField(desc="Explanation of the approach")
    code: str = dspy.OutputField(desc="The generated code")
    test_cases: str = dspy.OutputField(desc="Example test cases")


class SummarizationSignature(dspy.Signature):
    """Summarize text with key points."""

    text: str = dspy.InputField(desc="Text to summarize")
    length: str = dspy.InputField(desc="Desired length (e.g., 'brief', 'detailed')", default="brief")
    key_points: str = dspy.OutputField(desc="Key points from the text")
    summary: str = dspy.OutputField(desc="Concise summary")


class QuestionAnsweringSignature(dspy.Signature):
    """Answer questions about a given context."""

    context: str = dspy.InputField(desc="Context information")
    question: str = dspy.InputField(desc="Question to answer")
    answer: str = dspy.OutputField(desc="Answer based on the context")
    evidence: str = dspy.OutputField(desc="Evidence from context supporting the answer")


class ClassificationSignature(dspy.Signature):
    """Classify text into categories."""

    text: str = dspy.InputField(desc="Text to classify")
    categories: str = dspy.InputField(desc="Possible categories (comma-separated)")
    reasoning: str = dspy.OutputField(desc="Reasoning for the classification")
    classification: str = dspy.OutputField(desc="The chosen category")
    confidence: str = dspy.OutputField(desc="Confidence level (high/medium/low)")
