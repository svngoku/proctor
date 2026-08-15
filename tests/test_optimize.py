import asyncio
from unittest.mock import patch

import pytest

from proctor.optimize import (
    APEOptimizer,
    OPTIMIZERS,
    ProTeGiOptimizer,
    RLPromptOptimizer,
)
from proctor.optimize.ape import APEOptimizer as APE
from proctor.thought_generation.techniques import ChainOfThought


def test_evaluate_response():
    ape = APE()
    assert ape._evaluate_response("50", "50")
    assert ape._evaluate_response("The answer is 50", "50")
    assert ape._evaluate_response("1.00", "1")
    assert not ape._evaluate_response("nope", "50")


def test_registry_and_stubs():
    assert set(OPTIMIZERS) == {"ape", "protegi", "rlprompt"}
    with pytest.raises(NotImplementedError):
        asyncio.run(ProTeGiOptimizer().optimize("task", ["seed"], []))
    with pytest.raises(NotImplementedError):
        asyncio.run(RLPromptOptimizer().optimize("task", ["seed"], []))


@patch("proctor.optimize.ape.call_llm")
def test_ape_empty_candidates_fail(mock_call_llm):
    mock_call_llm.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(
            APEOptimizer().optimize("task", [], [{"input": "1", "output": "1"}])
        )


def test_ape_uses_custom_instructions():
    technique = ChainOfThought()
    prompt = technique.generate_prompt("2+2", custom_instructions="IGNORE PREVIOUS")
    assert "IGNORE PREVIOUS" in prompt
    assert technique.generate_prompt(
        "2+2", instruction="IGNORE"
    ) == technique.generate_prompt("2+2")
