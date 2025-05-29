"""
Unit tests for prompt techniques.
"""
import unittest
from typing import Dict

from proctor import (
    get_technique,
    list_techniques,
    ZeroShotCoT,
    ChainOfThought,
    RolePrompting,
    CompositeTechnique
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
            description="Role-based reasoning"
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


if __name__ == "__main__":
    unittest.main()