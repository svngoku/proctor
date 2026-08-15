"""
Unit tests for utility functions.
"""

import unittest
from unittest.mock import patch, MagicMock

import httpx
from openrouter.errors import OpenRouterDefaultError, BadRequestResponseError

from proctor.utils import (
    LLMError,
    call_llm,
    call_llm_async,
    call_llm_async_stream,
    call_llm_stream,
    dedent_prompt,
)


def _make_response(content):
    """Build a mock chat response with the given message content."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _retryable_error(message="Service unavailable"):
    """Build an OpenRouter error that the retry loop treats as transient."""
    return OpenRouterDefaultError(message, httpx.Response(503))


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""

    def test_dedent_prompt(self):
        """Test dedent_prompt function."""
        indented = """
            This is a test prompt
            with multiple lines
                and varying indentation
            levels.
        """

        expected = "This is a test prompt\nwith multiple lines\n    and varying indentation\nlevels."
        result = dedent_prompt(indented)
        self.assertEqual(result, expected)

        single_line = "Single line prompt"
        self.assertEqual(dedent_prompt(single_line), single_line)

        self.assertEqual(dedent_prompt(""), "")

    @patch("proctor.utils.OpenRouter")
    @patch("proctor.utils.get_llm_config")
    def test_call_llm_success(self, mock_get_config, mock_openrouter):
        """Test successful LLM call."""
        mock_get_config.return_value = {
            "model": "test-model",
            "api_key": "test-key",
        }

        client = mock_openrouter.return_value.__enter__.return_value
        client.chat.send.return_value = _make_response("Test response")

        result = call_llm("Test prompt")

        self.assertEqual(result, "Test response")
        client.chat.send.assert_called_once()

    @patch("proctor.utils.OpenRouter")
    @patch("proctor.utils.get_llm_config")
    def test_call_llm_invalid_input(self, mock_get_config, mock_openrouter):
        """Test LLM call with invalid input."""
        with self.assertRaises(ValueError):
            call_llm("")

        with self.assertRaises(ValueError):
            call_llm(123)  # type: ignore

        mock_openrouter.assert_not_called()

    @patch("proctor.utils.OpenRouter")
    @patch("proctor.utils.get_llm_config")
    def test_call_llm_missing_api_key(self, mock_get_config, mock_openrouter):
        """Test LLM call with missing API key."""
        mock_get_config.return_value = {
            "model": "test-model",
            "api_key": "",  # Empty API key
        }

        with self.assertRaises(LLMError) as context:
            call_llm("Test prompt")

        self.assertIn("Missing API key", str(context.exception))
        mock_openrouter.assert_not_called()

    @patch("proctor.utils.OpenRouter")
    @patch("proctor.utils.get_llm_config")
    @patch("time.sleep")  # Mock sleep to avoid delays in tests
    def test_call_llm_retry_success(
        self, mock_sleep, mock_get_config, mock_openrouter
    ):
        """Test LLM call with retry that eventually succeeds."""
        mock_get_config.return_value = {
            "model": "test-model",
            "api_key": "test-key",
        }

        client = mock_openrouter.return_value.__enter__.return_value
        client.chat.send.side_effect = [
            _retryable_error("Rate limit exceeded"),
            _make_response("Success after retry"),
        ]

        result = call_llm("Test prompt", max_retries=1)

        self.assertEqual(result, "Success after retry")
        self.assertEqual(client.chat.send.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("proctor.utils.OpenRouter")
    @patch("proctor.utils.get_llm_config")
    @patch("time.sleep")  # Mock sleep to avoid delays in tests
    def test_call_llm_max_retries_exceeded(
        self, mock_sleep, mock_get_config, mock_openrouter
    ):
        """Test LLM call with retries that all fail."""
        mock_get_config.return_value = {
            "model": "test-model",
            "api_key": "test-key",
        }

        error_msg = "Service unavailable"
        client = mock_openrouter.return_value.__enter__.return_value
        client.chat.send.side_effect = _retryable_error(error_msg)

        with self.assertRaises(LLMError) as context:
            call_llm("Test prompt", max_retries=2)

        self.assertIn(error_msg, str(context.exception))
        self.assertEqual(client.chat.send.call_count, 3)  # Initial call + 2 retries
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("proctor.utils.OpenRouter")
    @patch("proctor.utils.get_llm_config")
    def test_call_llm_non_retryable_error(self, mock_get_config, mock_openrouter):
        """Test that a 4xx client error is not retried."""
        mock_get_config.return_value = {
            "model": "test-model",
            "api_key": "test-key",
        }

        mock_openrouter.return_value.__exit__.return_value = False
        client = mock_openrouter.return_value.__enter__.return_value
        client.chat.send.side_effect = BadRequestResponseError(
            MagicMock(), httpx.Response(400)
        )

        with self.assertRaises(LLMError):
            call_llm("Test prompt", max_retries=2)

        client.chat.send.assert_called_once()

    @patch("proctor.utils.OpenRouter")
    @patch("proctor.utils.get_llm_config")
    def test_call_llm_passes_api_base(self, mock_get_config, mock_openrouter):
        mock_get_config.return_value = {
            "model": "test-model",
            "api_key": "test-key",
            "api_base": "https://proxy.example/v1",
        }
        client = mock_openrouter.return_value.__enter__.return_value
        client.chat.send.return_value = _make_response("ok")

        call_llm("Test prompt")

        mock_openrouter.assert_called_once_with(
            api_key="test-key", server_url="https://proxy.example/v1"
        )

    @patch("proctor.utils.OpenRouter")
    @patch("proctor.utils.get_llm_config")
    def test_call_llm_stream(self, mock_get_config, mock_openrouter):
        mock_get_config.return_value = {"model": "test-model", "api_key": "test-key"}
        event = MagicMock()
        event.choices = [MagicMock()]
        event.choices[0].delta.content = "chunk"
        client = mock_openrouter.return_value.__enter__.return_value
        client.chat.send.return_value = [event]

        self.assertEqual(list(call_llm_stream("Test prompt")), ["chunk"])


@patch("proctor.utils.OpenRouter")
@patch("proctor.utils.get_llm_config")
async def test_call_llm_async(mock_get_config, mock_openrouter):
    mock_get_config.return_value = {"model": "test-model", "api_key": "test-key"}
    client = mock_openrouter.return_value.__aenter__.return_value
    client.chat.send_async.return_value = _make_response("async-ok")

    result = await call_llm_async("Test prompt")

    assert result == "async-ok"
    client.chat.send_async.assert_called_once()


@patch("proctor.utils.OpenRouter")
@patch("proctor.utils.get_llm_config")
async def test_call_llm_async_stream(mock_get_config, mock_openrouter):
    mock_get_config.return_value = {"model": "test-model", "api_key": "test-key"}
    event = MagicMock()
    event.choices = [MagicMock()]
    event.choices[0].delta.content = "async-chunk"

    async def _events():
        yield event

    client = mock_openrouter.return_value.__aenter__.return_value
    client.chat.send_async.return_value = _events()

    chunks = [chunk async for chunk in call_llm_async_stream("Test prompt")]
    assert chunks == ["async-chunk"]


if __name__ == "__main__":
    unittest.main()
