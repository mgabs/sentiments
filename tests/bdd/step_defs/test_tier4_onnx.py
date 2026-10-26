import socket
import pytest
from unittest.mock import MagicMock, patch
from pytest_bdd import scenarios, given, when, then, parsers

from sentiments.core.models import Tier, SentimentLabel, SentimentResult
from sentiments.providers.base import ClassifierProviderError
from sentiments.providers.tier4.classifier import ONNXClassifier
from sentiments.providers.tier4.protocol import ONNXBackend
from sentiments.providers.tier4.mock import MockONNXBackend

scenarios("../features/tier4_onnx.feature")


@pytest.fixture
def tier4_state():
    return {
        "clf": None,
        "backend": None,
        "result": None,
        "results": None,
        "error": None,
    }


@given(
    parsers.parse(
        'an ONNX backend configured with model "{model}" returning "{label}" with confidence {confidence:f}'
    )
)
def given_onnx_backend(tier4_state, model, label, confidence):
    backend = MagicMock(spec=ONNXBackend)
    backend.model_name = model
    backend.is_available.return_value = True
    backend.predict.return_value = (SentimentLabel(label), confidence, {"provider": "mock"})
    tier4_state["backend"] = backend
    tier4_state["clf"] = ONNXClassifier(backend=backend)


@when(parsers.parse('the ONNXClassifier classifies "{text}"'))
def when_classify_single(tier4_state, text):
    try:
        tier4_state["result"] = tier4_state["clf"].classify(text)
    except Exception as e:
        tier4_state["error"] = e


@then(parsers.parse('the result has tier "{tier}"'))
def then_result_tier(tier4_state, tier):
    assert tier4_state["result"].tier == Tier(tier)


@then(parsers.parse('the result has model name "{model}"'))
def then_result_model(tier4_state, model):
    assert tier4_state["result"].model_name == model


@then(parsers.parse('the result has label "{label}"'))
def then_result_label(tier4_state, label):
    assert tier4_state["result"].label == SentimentLabel(label)


@then(parsers.parse("the result confidence is {confidence:f}"))
def then_result_confidence(tier4_state, confidence):
    assert tier4_state["result"].confidence == pytest.approx(confidence, rel=1e-3)


@then("the latency in milliseconds is positive")
def then_latency_positive(tier4_state):
    assert tier4_state["result"].latency_ms >= 0.0


@given(parsers.parse('an ONNX backend configured with model "{model}"'))
def given_onnx_backend_batch(tier4_state, model):
    backend = MockONNXBackend(model_name=model)
    tier4_state["backend"] = backend
    tier4_state["clf"] = ONNXClassifier(backend=backend)


@when("the ONNXClassifier classifies a batch of 3 texts")
def when_classify_batch_3(tier4_state):
    texts = ["Item is wonderful", "Terrible broke immediately", "Average plain normal"]
    tier4_state["results"] = tier4_state["clf"].classify_batch(texts)


@then("exactly 3 results are returned in matching order")
def then_batch_results_count(tier4_state):
    assert len(tier4_state["results"]) == 3


@then(parsers.parse('each result has tier "{tier}"'))
def then_each_result_tier(tier4_state, tier):
    for r in tier4_state["results"]:
        assert r.tier == Tier(tier)


@then("each result has valid sentiment labels and confidence scores")
def then_each_result_valid(tier4_state):
    for r in tier4_state["results"]:
        assert isinstance(r.label, SentimentLabel)
        assert 0.0 <= r.confidence <= 1.0


@given("a configured ONNXClassifier")
def given_configured_clf(tier4_state):
    tier4_state["backend"] = MockONNXBackend()
    tier4_state["clf"] = ONNXClassifier(backend=tier4_state["backend"])


@when("classifying an empty or whitespace string with ONNXClassifier")
def when_classify_empty(tier4_state):
    try:
        tier4_state["clf"].classify("   ")
    except Exception as e:
        tier4_state["error"] = e


@then(parsers.parse('it raises a ValueError with message containing "{msg}"'))
def then_raises_value_error(tier4_state, msg):
    assert isinstance(tier4_state["error"], ValueError)
    assert msg in str(tier4_state["error"])


@given("an ONNX backend that fails on specific text")
def given_failing_backend(tier4_state):
    backend = MagicMock(spec=ONNXBackend)
    backend.model_name = "failing-backend"
    backend.is_available.return_value = True

    def side_effect(text):
        if "bad" in text:
            raise RuntimeError("Engine failure")
        return SentimentLabel.POSITIVE, 0.90, {}

    backend.predict.side_effect = side_effect
    tier4_state["backend"] = backend
    tier4_state["clf"] = ONNXClassifier(backend=backend)


@when("the ONNXClassifier classifies a batch with fail_fast set to false")
def when_classify_fail_fast_false(tier4_state):
    texts = ["good item", "bad broken", "good another"]
    tier4_state["results"] = tier4_state["clf"].classify_batch(texts, fail_fast=False)


@then("failed samples return fallback NEUTRAL results with error metadata")
def then_failed_samples_fallback(tier4_state):
    results = tier4_state["results"]
    assert results[1].label == SentimentLabel.NEUTRAL
    assert results[1].confidence == 0.33
    assert "error" in results[1].metadata
    assert results[1].latency_ms == 0.0


@then("valid samples return successful classifications")
def then_valid_samples_success(tier4_state):
    results = tier4_state["results"]
    assert results[0].label == SentimentLabel.POSITIVE
    assert results[2].label == SentimentLabel.POSITIVE


@when("the ONNXClassifier classifies a batch with fail_fast set to true")
def when_classify_fail_fast_true(tier4_state):
    texts = ["good item", "bad broken"]
    try:
        tier4_state["clf"].classify_batch(texts, fail_fast=True)
    except Exception as e:
        tier4_state["error"] = e


@then("it raises a ClassifierProviderError immediately")
def then_raises_provider_error(tier4_state):
    assert isinstance(tier4_state["error"], ClassifierProviderError)


@given("an initialized ONNXClassifier")
def given_initialized_clf(tier4_state):
    tier4_state["backend"] = MockONNXBackend()
    tier4_state["clf"] = ONNXClassifier(backend=tier4_state["backend"])


@when("classifying text while all network sockets are blocked")
def when_blocked_socket(tier4_state):
    def blocked_connect(*args, **kwargs):
        raise AssertionError("Network socket connection attempted during ONNX inference!")

    with patch.object(socket.socket, "connect", side_effect=blocked_connect):
        tier4_state["result"] = tier4_state["clf"].classify("Fast and clean transaction.")


@then("classification succeeds without initiating any network connections")
def then_success_no_network(tier4_state):
    assert tier4_state["result"] is not None
    assert isinstance(tier4_state["result"].label, SentimentLabel)
