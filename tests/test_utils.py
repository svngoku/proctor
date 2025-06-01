"""
Unit tests for utility functions.
"""

import unittest
from unittest.mock import patch, MagicMock

from proctor.utils import dedent_prompt, call_llm, LLMError


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""

    def test_dedent_prompt(self):
        """Test dedent_prompt function."""
        # Test with indented multi-line string
        indented = """
            This is a test prompt
            with multiple lines
                and varying indentation
            levels.
        """

        expected = "This is a test prompt\nwith multiple lines\n    and varying indentation\nlevels."
        result = dedent_prompt(indented)
        self.assertEqual(result, expected)

        # Test with single line
        single_line = "Single line prompt"
        self.assertEqual(dedent_prompt(single_line), single_line)

        # Test with empty string
        self.assertEqual(dedent_prompt(""), "")

    @patch("proctor.utils.litellm.completion")
    @patch("proctor.utils.get_llm_config")
    def test_call_llm_success(self, mock_get_config, mock_completion):
        """Test successful LLM call."""
        # Mock configuration
        mock_get_config.return_value = {
            "model": "test-model",
            "api_base": "https://api.test.com",
            "api_key": "test-key",
        }

        # Mock successful response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_completion.return_value = mock_response

        # Call function
        result = call_llm("Test prompt")

        # Verify
        self.assertEqual(result, "Test response")
        mock_completion.assert_called_once()

    @patch("proctor.utils.litellm.completion")
    @patch("proctor.utils.get_llm_config")
    def test_call_llm_invalid_input(self, mock_get_config, mock_completion):
        """Test LLM call with invalid input."""
        # Empty prompt should raise ValueError
        with self.assertRaises(ValueError):
            call_llm("")

        # Non-string prompt should raise ValueError
        with self.assertRaises(ValueError):
            call_llm(123)  # type: ignore

        # No calls to litellm should be made
        mock_completion.assert_not_called()

    @patch("proctor.utils.litellm.completion")
    @patch("proctor.utils.get_llm_config")
    def test_call_llm_missing_api_key(self, mock_get_config, mock_completion):
        """Test LLM call with missing API key."""
        # Mock configuration without API key
        mock_get_config.return_value = {
            "model": "test-model",
            "api_base": "https://api.test.com",
            "api_key": "",  # Empty API key
        }

        # Should raise LLMError
        with self.assertRaises(LLMError) as context:
            call_llm("Test prompt")

        # Verify error message
        self.assertIn("Missing API key", str(context.exception))

        # No calls to litellm should be made
        mock_completion.assert_not_called()

    @patch("proctor.utils.litellm.completion")
    @patch("proctor.utils.get_llm_config")
    @patch("time.sleep")  # Mock sleep to avoid delays in tests
    def test_call_llm_retry_success(self, mock_sleep, mock_get_config, mock_completion):
        """Test LLM call with retry that eventually succeeds."""
        # Import here to avoid circular imports
        import litellm.exceptions

        # Mock configuration
        mock_get_config.return_value = {
            "model": "test-model",
            "api_base": "https://api.test.com",
            "api_key": "test-key",
        }

        # Set up a custom rate limit error for the first call
        rate_limit_error = litellm.exceptions.RateLimitError(
            message="Rate limit exceeded",
            llm_provider="test-provider",
            model="test-model",
        )

        # Set up successful response for second call
        success_response = MagicMock()
        success_response.choices = [MagicMock()]
        success_response.choices[0].message.content = "Success after retry"

        # Set side effect for mock_completion
        mock_completion.side_effect = [
            rate_limit_error,  # First call fails with rate limit error
            success_response,  # Second call succeeds
        ]

        # Call function with max_retries=1
        result = call_llm("Test prompt", max_retries=1)

        # Verify
        self.assertEqual(result, "Success after retry")
        self.assertEqual(mock_completion.call_count, 2)  # Should be called twice
        mock_sleep.assert_called_once()  # Should sleep once between retries

    @patch("proctor.utils.litellm.completion")
    @patch("proctor.utils.get_llm_config")
    @patch("time.sleep")  # Mock sleep to avoid delays in tests
    def test_call_llm_max_retries_exceeded(
        self, mock_sleep, mock_get_config, mock_completion
    ):
        """Test LLM call with retries that all fail."""
        # Import here to avoid circular imports
        import litellm.exceptions

        # Mock configuration
        mock_get_config.return_value = {
            "model": "test-model",
            "api_base": "https://api.test.com",
            "api_key": "test-key",
        }

        # Create a custom error that's in the retryable exceptions list
        error_msg = "Service unavailable"
        # The ServiceUnavailableError constructor requires additional arguments in newer versions
        retryable_error = litellm.exceptions.ServiceUnavailableError(
            message=error_msg, llm_provider="test-provider", model="test-model"
        )
        mock_completion.side_effect = retryable_error

        # Call function with max_retries=2
        with self.assertRaises(LLMError) as context:
            call_llm("Test prompt", max_retries=2)

        # Verify error message and number of calls
        self.assertIn(error_msg, str(context.exception))
        self.assertEqual(mock_completion.call_count, 3)  # Initial call + 2 retries
        self.assertEqual(mock_sleep.call_count, 2)  # Should sleep twice between retries


if __name__ == "__main__":
    unittest.main()
