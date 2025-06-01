"""
Unit tests for prompt techniques.
"""

import unittest
import random

from proctor import (
    get_technique,
    list_techniques,
    ZeroShotCoT,
    ChainOfThought,
    RolePrompting,
    CompositeTechnique,
    EmotionPrompting,
    SelfAsk,
    ExampleGeneration,
    KNN,
    ChainOfVerification,
    DECOMP,
    SelfConsistency,
    StylePrompting,
)


class TestTechniques(unittest.TestCase):
    """Test cases for prompt techniques."""

    def test_get_technique(self):
        """Test retrieving techniques by name."""
        # Test getting a valid technique
        cot = get_technique("chain_of_thought")
        self.assertIsNotNone(cot)
        self.assertEqual(cot.name, "Chain-of-Thought")
        self.assertEqual(cot.identifier, "2.2.3")

        # Test getting an invalid technique
        invalid = get_technique("nonexistent_technique")
        self.assertIsNone(invalid)

        # Test caching - should return the same instance
        cot2 = get_technique("chain_of_thought")
        self.assertIs(cot, cot2)  # Should be the same object in memory

    def test_list_techniques(self):
        """Test listing available techniques."""
        # List all techniques
        all_techniques = list_techniques()
        self.assertIsInstance(all_techniques, list)
        self.assertIn("chain_of_thought", all_techniques)
        self.assertIn("zero_shot_cot", all_techniques)
        self.assertIn("role_prompting", all_techniques)

        # List techniques by category
        # Get techniques with identifier starting with "2.2.3"
        cot_techniques = list_techniques(category="2.2.3")
        self.assertIn("chain_of_thought", cot_techniques)
        self.assertIn("zero_shot_cot", cot_techniques)

        # Category with no techniques
        empty_category = list_techniques(category="nonexistent")
        self.assertEqual(empty_category, [])

    def test_zero_shot_cot(self):
        """Test ZeroShotCoT technique."""
        technique = ZeroShotCoT()
        self.assertEqual(technique.name, "Zero-Shot CoT")
        self.assertEqual(technique.identifier, "2.2.3.1")

        # Test prompt generation
        input_text = "What is the result of 25 × 4?"
        prompt = technique.generate_prompt(input_text)

        self.assertIn(input_text, prompt)
        self.assertIn("Let's think step by step", prompt)

        # Test with custom instructions
        custom_instr = "I'll solve this carefully:"
        prompt = technique.generate_prompt(input_text, custom_instructions=custom_instr)
        self.assertIn(custom_instr, prompt)

        # Test with invalid input
        with self.assertRaises(ValueError):
            technique.generate_prompt("")

    def test_chain_of_thought(self):
        """Test ChainOfThought technique."""
        technique = ChainOfThought(num_steps=3)
        self.assertEqual(technique.name, "Chain-of-Thought")

        # Test prompt generation
        input_text = "What is the result of 25 × 4?"
        prompt = technique.generate_prompt(input_text)

        self.assertIn(input_text, prompt)
        self.assertIn("1. ", prompt)
        self.assertIn("2. ", prompt)
        self.assertIn("3. ", prompt)
        self.assertIn("Therefore, the final answer is:", prompt)

        # Test with different number of steps
        technique = ChainOfThought(num_steps=2)
        prompt = technique.generate_prompt(input_text)
        self.assertIn("1. ", prompt)
        self.assertIn("2. ", prompt)
        self.assertNotIn("3. ", prompt)

    def test_composite_technique(self):
        """Test composite technique."""
        # Create a composite technique with RolePrompting and ZeroShotCoT
        composite = CompositeTechnique(
            name="Expert CoT",
            identifier="custom-composite",
            techniques=[RolePrompting(), ZeroShotCoT()],
            description="Role-based reasoning",
        )

        # Test properties
        self.assertEqual(composite.name, "Expert CoT")
        self.assertEqual(composite.identifier, "custom-composite")
        self.assertEqual(len(composite.techniques), 2)

        # Test prompt generation
        input_text = "Explain quantum computing."
        prompt = composite.generate_prompt(input_text, role="quantum physicist")

        # Should contain elements from both techniques
        self.assertIn("quantum physicist", prompt)
        self.assertIn("Let's think step by step", prompt)

    def test_emotion_prompting(self):
        """Test EmotionPrompting technique."""
        technique = EmotionPrompting()
        self.assertEqual(technique.name, "Emotion Prompting")
        self.assertEqual(technique.identifier, "2.2.2.1")

        # Test prompt generation with default emotion
        input_text = "Explain the solar system."
        prompt = technique.generate_prompt(input_text)

        self.assertIn(input_text, prompt)
        self.assertIn("excited", prompt)

        # Test with custom emotion
        prompt = technique.generate_prompt(input_text, emotion="curious")
        self.assertIn("curious", prompt)

        # Test with intensity and context
        prompt = technique.generate_prompt(
            input_text,
            emotion="enthusiastic",
            intensity="very",
            context="You're teaching astronomy to beginners.",
        )

        self.assertIn("very enthusiastic", prompt)
        self.assertIn("You're teaching astronomy to beginners.", prompt)

    def test_role_prompting(self):
        """Test RolePrompting technique."""
        technique = RolePrompting()
        self.assertEqual(technique.name, "Role Prompting")
        self.assertEqual(technique.identifier, "2.2.2.2")

        # Test prompt generation with default role
        input_text = "Explain quantum entanglement."
        prompt = technique.generate_prompt(input_text)

        self.assertIn(input_text, prompt)
        self.assertIn("expert", prompt)

        # Test with custom role
        prompt = technique.generate_prompt(input_text, role="physicist")
        self.assertIn("physicist", prompt)

        # Test with field, experience, and audience
        prompt = technique.generate_prompt(
            input_text,
            role="physicist",
            field="quantum mechanics",
            experience="renowned",
            audience="university students",
        )

        self.assertIn("renowned physicist", prompt)
        self.assertIn("quantum mechanics", prompt)
        self.assertIn("Your target audience is university students", prompt)

    def test_style_prompting(self):
        """Test StylePrompting technique."""
        technique = StylePrompting()
        self.assertEqual(technique.name, "Style Prompting")
        self.assertEqual(technique.identifier, "2.2.2.3")

        # Test prompt generation with default style
        input_text = "Explain the concept of relativity."
        prompt = technique.generate_prompt(input_text)

        self.assertIn(input_text, prompt)
        self.assertIn("professional style", prompt)

        # Test with custom style
        prompt = technique.generate_prompt(input_text, style="casual")
        self.assertIn("casual style", prompt)

        # Test with tone, format, and audience
        prompt = technique.generate_prompt(
            input_text,
            style="academic",
            tone="formal",
            format="essay",
            audience="undergraduates",
        )

        self.assertIn("academic style", prompt)
        self.assertIn("with a formal tone", prompt)
        self.assertIn("in essay format", prompt)
        self.assertIn("for undergraduates", prompt)

    def test_self_ask(self):
        """Test SelfAsk technique."""
        technique = SelfAsk()
        self.assertEqual(technique.name, "Self-Ask")
        self.assertEqual(technique.identifier, "2.2.2.8")

        # Test prompt generation with defaults
        input_text = "What causes ocean tides?"
        prompt = technique.generate_prompt(input_text)

        self.assertIn(input_text, prompt)
        self.assertIn("Main Question:", prompt)
        self.assertIn("self-questioning approach", prompt)

        # Check default number of questions
        self.assertEqual(prompt.count("[Ask a specific"), 3)

        # Test with custom number of questions
        prompt = technique.generate_prompt(input_text, num_questions=2)
        self.assertEqual(prompt.count("[Ask a specific"), 2)

        # Test with domain and depth
        prompt = technique.generate_prompt(
            input_text, domain="astrophysics", depth="deep"
        )

        self.assertIn("in the domain of astrophysics", prompt)
        self.assertIn("delve into underlying principles", prompt)

    def test_example_generation(self):
        """Test ExampleGeneration technique."""
        technique = ExampleGeneration()
        self.assertEqual(technique.name, "Example Generation")
        self.assertEqual(technique.identifier, "2.2.1.1")

        # Test prompt generation with default examples
        input_text = "Classify this sentiment."
        prompt = technique.generate_prompt(input_text)

        self.assertIn(input_text, prompt)
        self.assertIn("Example input 1", prompt)
        self.assertIn("Example output 1", prompt)
        self.assertEqual(prompt.count("Input:"), 4)  # 3 examples + 1 actual input

        # Test with custom examples
        custom_examples = [
            {"input": "The movie was terrible.", "output": "Negative"},
            {"input": "I loved the concert!", "output": "Positive"},
        ]

        prompt = technique.generate_prompt(input_text, examples=custom_examples)
        self.assertIn("The movie was terrible.", prompt)
        self.assertIn("Negative", prompt)
        self.assertIn("I loved the concert!", prompt)
        self.assertIn("Positive", prompt)
        self.assertEqual(prompt.count("Input:"), 3)  # 2 examples + 1 actual input

    def test_knn(self):
        """Test KNN technique."""
        technique = KNN()
        self.assertEqual(technique.name, "KNN")
        self.assertEqual(technique.identifier, "2.2.1.2")

        # Test prompt generation with empty examples pool
        input_text = "Explain neural networks."
        prompt = technique.generate_prompt(input_text)

        self.assertIn(input_text, prompt)
        self.assertIn("[No similar examples found]", prompt)

        # Test with examples pool
        examples_pool = [
            {"input": "What is machine learning?", "output": "Machine learning is..."},
            {"input": "How do neural networks work?", "output": "Neural networks..."},
            {"input": "Explain deep learning.", "output": "Deep learning is..."},
        ]

        # Mock random.sample to return predictable results for testing
        original_sample = random.sample
        try:
            random.sample = lambda population, k: population[:k]

            prompt = technique.generate_prompt(
                input_text, examples_pool=examples_pool, k=2
            )
            self.assertIn("What is machine learning?", prompt)
            self.assertIn("How do neural networks work?", prompt)
            self.assertNotIn("Explain deep learning.", prompt)
            self.assertEqual(prompt.count("Input:"), 3)  # 2 examples + 1 actual input

        finally:
            # Restore original random.sample
            random.sample = original_sample

    def test_chain_of_verification(self):
        """Test ChainOfVerification technique."""
        technique = ChainOfVerification()
        self.assertEqual(technique.name, "Chain-of-Verification")
        self.assertEqual(technique.identifier, "2.2.6")

        # Test prompt generation with defaults
        input_text = "What is the capital of France?"
        prompt = technique.generate_prompt(input_text)

        self.assertIn(input_text, prompt)
        self.assertIn("Problem Statement:", prompt)
        self.assertIn("Verification Approach:", prompt)
        self.assertIn("Initial Solution:", prompt)
        self.assertIn("Verification of Step", prompt)

        # Check default verification aspects
        self.assertIn("factual correctness", prompt)
        self.assertIn("logical consistency", prompt)

        # Test with custom number of steps and verification aspects
        custom_aspects = ["mathematical accuracy", "conceptual clarity"]
        prompt = technique.generate_prompt(
            input_text,
            num_steps=2,
            verification_aspects=custom_aspects,
            verification_intensity="rigorous",
            include_counterexamples=True,
        )

        self.assertEqual(prompt.count("Verification of Step"), 2)
        self.assertIn("mathematical accuracy", prompt)
        self.assertIn("conceptual clarity", prompt)
        self.assertIn("construct counterexamples", prompt)
        self.assertIn("exhaustive verification", prompt)

        # Test with domain
        prompt = technique.generate_prompt(input_text, domain="geography")
        self.assertIn("in the geography context", prompt)

    def test_decomp(self):
        """Test DECOMP technique."""
        technique = DECOMP()
        self.assertEqual(technique.name, "DECOMP")
        self.assertEqual(technique.identifier, "2.2.4")

        # Test prompt generation with defaults
        input_text = "How do trees contribute to the ecosystem?"
        prompt = technique.generate_prompt(input_text)

        self.assertIn(input_text, prompt)
        self.assertIn("Complex Problem Analysis:", prompt)
        self.assertIn("Decomposition Strategy:", prompt)
        self.assertIn("Breaking Down the Problem:", prompt)

        # Check default number of subproblems
        self.assertEqual(prompt.count("Subproblem"), 6)

        # Test with custom parameters
        prompt = technique.generate_prompt(
            input_text,
            num_subproblems=2,
            approach="hierarchical",
            domain="ecology",
            clear_dependencies=True,
        )

        self.assertEqual(prompt.count("Subproblem"), 4)
        self.assertIn("in the ecology domain", prompt)
        self.assertIn("Break the problem into major components", prompt)
        self.assertIn("Explicitly note how each subproblem depends on", prompt)

    def test_self_consistency(self):
        """Test SelfConsistency technique."""
        technique = SelfConsistency()
        self.assertEqual(technique.name, "Self-Consistency")
        self.assertEqual(technique.identifier, "2.2.5")

        # Test prompt generation with defaults
        input_text = "What is the best way to learn programming?"
        prompt = technique.generate_prompt(input_text)

        self.assertIn(input_text, prompt)
        self.assertIn("Multiple-Path Problem Solving", prompt)
        self.assertIn("Independent Reasoning Paths:", prompt)
        self.assertIn("Analysis of Results:", prompt)
        self.assertIn("Consensus Determination:", prompt)

        # Check default number of paths
        self.assertEqual(prompt.count("Path "), 5)

        # Test with custom parameters
        prompt = technique.generate_prompt(
            input_text,
            num_paths=2,
            approach_diversity=True,
            domain="computer science",
            include_metacognition=True,
            path_length="detailed",
        )

        self.assertEqual(prompt.count("Path "), 4)
        self.assertIn("in the computer science domain", prompt)
        self.assertIn("Elaborate thoroughly on each reasoning path", prompt)
        self.assertIn("briefly note your confidence level", prompt)

        # Test with reasoning styles
        reasoning_styles = ["analytical", "empirical"]
        prompt = technique.generate_prompt(
            input_text, num_paths=2, reasoning_styles=reasoning_styles
        )

        self.assertIn("Path 1 using analytical reasoning", prompt)
        self.assertIn("Path 2 using empirical reasoning", prompt)


if __name__ == "__main__":
    unittest.main()
