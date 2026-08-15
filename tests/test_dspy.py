from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import dspy
import pytest

from proctor.dspy_base import DSPyCompositeTechnique, DSPyPromptTechnique
from proctor.dspy_lm import LiteLLMLanguageModel
from proctor.dspy_techniques.thought_generation import DSPySelfConsistency
from proctor.thought_generation.techniques import ChainOfThought
from proctor.utils import LLMError


class _Dummy(DSPyPromptTechnique):
    def forward(self, **kwargs):
        return dspy.Prediction(reasoning="r", answer="a")


def test_execute_uses_prediction_fields():
    assert "reasoning: r" in _Dummy("n", "id").execute("q").lower()


def test_composite_forwards_prediction_fields():
    class Second(DSPyPromptTechnique):
        def forward(self, **kwargs):
            assert kwargs["answer"] == "a"
            return dspy.Prediction(answer=kwargs["answer"])

    result = DSPyCompositeTechnique(
        "c", "id", modules=[_Dummy("n", "id"), Second("s", "sid")]
    ).forward(problem="q")
    assert result.answer == "a"


def test_self_consistency_vote_and_all_failed():
    sc = DSPySelfConsistency(num_paths=3)
    good = SimpleNamespace(answer="9", reasoning="left")
    sc.predictors = [
        lambda **_: good,
        lambda **_: good,
        lambda **_: SimpleNamespace(answer="0", reasoning="x"),
    ]
    result = sc.forward(problem="q")
    assert result.answer == "9"
    assert result.confidence == pytest.approx(2 / 3)

    sc.predictors = [lambda **_: (_ for _ in ()).throw(RuntimeError("fail"))] * 3
    with pytest.raises(RuntimeError, match="All reasoning paths failed"):
        sc.forward(problem="q")


@patch("proctor.dspy_lm.call_llm", return_value="ok")
@patch(
    "proctor.dspy_lm.get_llm_config",
    return_value={"model": "m", "api_key": "k", "api_base": "https://x"},
)
def test_lm_forward_uses_call_llm(mock_cfg, mock_call):
    lm = LiteLLMLanguageModel()
    out = lm.forward(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert out.choices[0].message.content == "ok"
    mock_call.assert_called_once()
    assert mock_call.call_args.kwargs["system_prompt"] == "sys"
    assert hasattr(lm, "aforward")


@patch("proctor.base.call_llm_async", new_callable=AsyncMock, return_value="async-ok")
async def test_execute_async(mock_async):
    result = await ChainOfThought().execute_async("2+2")
    assert result == "async-ok"
    mock_async.assert_awaited_once()


@patch(
    "proctor.base.call_llm_async", new_callable=AsyncMock, side_effect=LLMError("nope")
)
async def test_execute_async_reraises_llm_error(mock_async):
    with pytest.raises(LLMError):
        await ChainOfThought().execute_async("2+2")
