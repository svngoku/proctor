"""
Unit tests for configuration module.
"""

import unittest
from unittest.mock import patch
import os
from proctor.config import get_llm_config, DEFAULT_LLM_CONFIG


class TestConfig(unittest.TestCase):
    """Test cases for configuration functions."""

    def test_default_config(self):
        """Test that DEFAULT_LLM_CONFIG has expected values."""
        self.assertEqual(DEFAULT_LLM_CONFIG["model"], "openai/gpt-4o")
        self.assertEqual(DEFAULT_LLM_CONFIG["api_base"], "https://openrouter.ai/api/v1")
        self.assertIn("api_key", DEFAULT_LLM_CONFIG)
        self.assertEqual(DEFAULT_LLM_CONFIG["max_tokens"], 1000)
        self.assertEqual(DEFAULT_LLM_CONFIG["temperature"], 0.7)

    def test_get_llm_config_default(self):
        """Test get_llm_config with no environment variables."""
        # Mock environment with no relevant variables set
        with patch.dict(os.environ, {}, clear=True):
            config = get_llm_config()

            # Should return a copy of the default config
            self.assertEqual(config, DEFAULT_LLM_CONFIG)

            # Should be a different object (copy, not reference)
            self.assertIsNot(config, DEFAULT_LLM_CONFIG)

    def test_get_llm_config_with_api_key(self):
        """Test get_llm_config with API key in environment."""
        # Mock environment with API key set
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-123"}, clear=True):
            config = get_llm_config()

            # Should use the environment variable value
            self.assertEqual(config["api_key"], "test-key-123")

            # Other values should remain unchanged
            self.assertEqual(config["model"], DEFAULT_LLM_CONFIG["model"])
            self.assertEqual(config["api_base"], DEFAULT_LLM_CONFIG["api_base"])

    def test_get_llm_config_with_model(self):
        """Test get_llm_config with model in environment."""
        # Mock environment with model set
        with patch.dict(
            os.environ,
            {"OPENROUTER_MODEL": "anthropic/claude-3-opus-20240229"},
            clear=True,
        ):
            config = get_llm_config()

            # Should use the environment variable value
            self.assertEqual(config["model"], "anthropic/claude-3-opus-20240229")

            # Other values should remain unchanged
            self.assertEqual(config["api_key"], DEFAULT_LLM_CONFIG["api_key"])
            self.assertEqual(config["api_base"], DEFAULT_LLM_CONFIG["api_base"])

    def test_get_llm_config_with_all_env_vars(self):
        """Test get_llm_config with all environment variables set."""
        # Mock environment with all variables set
        env_vars = {
            "OPENROUTER_API_KEY": "test-key-456",
            "OPENROUTER_MODEL": "anthropic/claude-3-sonnet-20240229",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = get_llm_config()

            # Should use all environment variable values
            self.assertEqual(config["api_key"], "test-key-456")
            self.assertEqual(config["model"], "anthropic/claude-3-sonnet-20240229")

            # Non-overridden values should remain unchanged
            self.assertEqual(config["api_base"], DEFAULT_LLM_CONFIG["api_base"])
            self.assertEqual(config["max_tokens"], DEFAULT_LLM_CONFIG["max_tokens"])
            self.assertEqual(config["temperature"], DEFAULT_LLM_CONFIG["temperature"])


if __name__ == "__main__":
    unittest.main()
