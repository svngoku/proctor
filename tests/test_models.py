from proctor.models import (
    ALL_OPENROUTER_MODELS,
    FLOATING_ALIASES,
    is_known_model,
    list_models,
    list_open_source_models,
)


def test_list_models_all_and_provider():
    assert list_models() == ALL_OPENROUTER_MODELS
    assert list_models("openai")
    assert list_models("no-such-provider") == []


def test_list_open_source_models():
    assert list_open_source_models()
    assert list_open_source_models("no-such-family") == []


def test_is_known_model():
    assert is_known_model("openai/gpt-4o")
    assert is_known_model("openai-latest")
    assert is_known_model(next(iter(FLOATING_ALIASES.values())))
    assert not is_known_model("not-a-real-model")


if __name__ == "__main__":
    test_list_models_all_and_provider()
    test_list_open_source_models()
    test_is_known_model()
    print("ok")
