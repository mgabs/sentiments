import pytest
from unittest.mock import patch, MagicMock
from pytest_bdd import scenarios, given, when, then, parsers

from sentiments.core.models import Tier, SentimentLabel
from sentiments.providers.base import ClassifierProviderError
from sentiments.providers.tier2.classifier import Tier2SLMClassifier
from sentiments.providers.tier2.ollama import OllamaBackend
from sentiments.providers.tier2.mlx import MLXBackend

scenarios("../features/tier2_slm.feature")


@pytest.fixture
def tier2_state():
    return {
        "clf": None,
        "backend": None,
        "result": None,
        "error": None,
        "backend_available": None,
    }


@given(
    parsers.parse(
        'an Ollama backend configured with model "{model}" returning valid JSON "{label}" with confidence {confidence:f}'
    )
)
def given_ollama_mock(tier2_state, model, label, confidence):
    backend = MagicMock(spec=OllamaBackend)
    backend.model_name = model
    backend.generate.return_value = f'{{"label": "{label}", "confidence": {confidence}}}'
    tier2_state["backend"] = backend
    tier2_state["clf"] = Tier2SLMClassifier(backend=backend)


@given(
    parsers.parse(
        'an MLX backend configured with model "{model}" returning valid JSON "{label}" with confidence {confidence:f}'
    )
)
def given_mlx_mock(tier2_state, model, label, confidence):
    backend = MagicMock(spec=MLXBackend)
    backend.model_name = model
    backend.generate.return_value = f'{{"label": "{label}", "confidence": {confidence}}}'
    tier2_state["backend"] = backend
    tier2_state["clf"] = Tier2SLMClassifier(backend=backend)


@given("an Ollama backend pointing to an unreachable port")
def given_ollama_unreachable(tier2_state):
    backend = OllamaBackend(model_name="llama3.2:3b", base_url="http://127.0.0.1:1")
    tier2_state["backend"] = backend
    tier2_state["clf"] = Tier2SLMClassifier(backend=backend)


@given("an MLX backend on an unsupported platform or missing mlx-lm library")
def given_mlx_unsupported(tier2_state):
    backend = MLXBackend()
    tier2_state["backend"] = backend
    tier2_state["clf"] = Tier2SLMClassifier(backend=backend)


@given("a backend returning markdown fenced JSON with extra commentary")
def given_fenced_json_backend(tier2_state):
    backend = MagicMock()
    backend.model_name = "llama3.2:3b"
    backend.generate.return_value = (
        "Analysis:\n```json\n"
        '{"label": "NEUTRAL", "confidence": 0.82, "explanation": "Factual review"}\n'
        "```\nThank you!"
    )
    tier2_state["backend"] = backend
    tier2_state["clf"] = Tier2SLMClassifier(backend=backend)


@given("a configured Tier2SLMClassifier")
def given_configured_clf(tier2_state):
    backend = MagicMock()
    backend.model_name = "llama3.2:3b"
    tier2_state["backend"] = backend
    tier2_state["clf"] = Tier2SLMClassifier(backend=backend)


@when(parsers.parse('the Tier2SLMClassifier classifies "{text}"'))
def when_classify_text(tier2_state, text):
    tier2_state["result"] = tier2_state["clf"].classify(text)


@when("the Tier2SLMClassifier attempts classification")
def when_attempt_classification(tier2_state):
    try:
        tier2_state["result"] = tier2_state["clf"].classify("Good product")
    except Exception as exc:
        tier2_state["error"] = exc


@when("checking backend availability or attempting generation")
def when_check_mlx_availability(tier2_state):
    with patch("sys.platform", "linux"):
        tier2_state["backend_available"] = tier2_state["backend"].is_available()
        try:
            tier2_state["clf"].classify("Sample text")
        except Exception as exc:
            tier2_state["error"] = exc


@when("classifying an empty or whitespace string")
def when_classify_whitespace(tier2_state):
    try:
        tier2_state["clf"].classify("   \n\t  ")
    except Exception as exc:
        tier2_state["error"] = exc


@then(parsers.parse('the result has tier "{tier}"'))
def then_tier(tier2_state, tier):
    assert tier2_state["result"].tier == Tier(tier)


@then(parsers.parse('the result has model name "{model}"'))
def then_model_name(tier2_state, model):
    assert tier2_state["result"].model_name == model


@then(parsers.parse('the result has label "{label}"'))
def then_label(tier2_state, label):
    assert tier2_state["result"].label == SentimentLabel(label)


@then(parsers.parse("the result confidence is {confidence:f}"))
def then_confidence(tier2_state, confidence):
    assert tier2_state["result"].confidence == pytest.approx(confidence)


@then("the latency in milliseconds is positive")
def then_latency(tier2_state):
    assert tier2_state["result"].latency_ms > 0.0


@then("it raises a ClassifierProviderError diagnosing unreachable Ollama daemon")
def then_unreachable_error(tier2_state):
    assert isinstance(tier2_state["error"], ClassifierProviderError)
    assert "unreachable" in str(tier2_state["error"]).lower() or "ollama" in str(tier2_state["error"]).lower()


@then("it indicates backend is unavailable or raises ClassifierProviderError")
def then_mlx_unavailable(tier2_state):
    assert tier2_state["backend_available"] is False
    assert isinstance(tier2_state["error"], ClassifierProviderError)


@then("the result confidence is clamped between 0.0 and 1.0")
def then_clamped(tier2_state):
    assert 0.0 <= tier2_state["result"].confidence <= 1.0


@then("it raises a ValueError before invoking the backend")
def then_value_error(tier2_state):
    assert isinstance(tier2_state["error"], ValueError)
    assert tier2_state["backend"].generate.call_count == 0
