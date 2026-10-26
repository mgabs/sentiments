import pytest
from unittest.mock import MagicMock

from sentiments.core.models import Tier, SentimentLabel, SentimentResult
from sentiments.providers.base import ClassifierProviderError
from sentiments.providers.tier4.classifier import ONNXClassifier
from sentiments.providers.tier4.protocol import ONNXBackend
from sentiments.providers.tier4.mock import MockONNXBackend


def test_onnx_classifier_properties():
    backend = MockONNXBackend(model_name="test-model")
    clf = ONNXClassifier(backend=backend)
    assert clf.tier == Tier.TIER_4
    assert clf.model_name == "test-model"


def test_onnx_classifier_empty_text_raises_value_error():
    clf = ONNXClassifier(backend=MockONNXBackend())
    with pytest.raises(ValueError, match="text must be a non-empty string"):
        clf.classify("")
    with pytest.raises(ValueError, match="text must be a non-empty string"):
        clf.classify("   ")
    with pytest.raises(ValueError, match="text must be a string"):
        clf.classify(None)  # type: ignore


def test_onnx_classifier_classify_single_success():
    backend = MagicMock(spec=ONNXBackend)
    backend.model_name = "minilm-int8"
    backend.is_available.return_value = True
    backend.predict.return_value = (SentimentLabel.POSITIVE, 0.96, {"test": True})

    clf = ONNXClassifier(backend=backend)
    res = clf.classify("Amazing service!")

    assert isinstance(res, SentimentResult)
    assert res.label == SentimentLabel.POSITIVE
    assert res.confidence == 0.96
    assert res.tier == Tier.TIER_4
    assert res.model_name == "minilm-int8"
    assert res.latency_ms >= 0.0
    assert res.metadata == {"test": True}


def test_onnx_classifier_classify_single_backend_error_wraps():
    backend = MagicMock(spec=ONNXBackend)
    backend.model_name = "minilm-int8"
    backend.predict.side_effect = RuntimeError("Crash in ONNX")

    clf = ONNXClassifier(backend=backend)
    with pytest.raises(ClassifierProviderError, match="ONNX inference failed: Crash in ONNX"):
        clf.classify("Some input")


def test_onnx_classifier_classify_single_provider_error_reraises():
    backend = MagicMock(spec=ONNXBackend)
    backend.model_name = "minilm-int8"
    backend.predict.side_effect = ClassifierProviderError("Custom provider failure")

    clf = ONNXClassifier(backend=backend)
    with pytest.raises(ClassifierProviderError, match="Custom provider failure"):
        clf.classify("Some input")


def test_onnx_classifier_classify_batch_resilient():
    backend = MagicMock(spec=ONNXBackend)
    backend.model_name = "minilm-int8"

    def fake_predict(text):
        if text == "fail":
            raise ValueError("Corrupt token")
        return SentimentLabel.NEGATIVE, 0.88, {}

    backend.predict.side_effect = fake_predict

    clf = ONNXClassifier(backend=backend)
    results = clf.classify_batch(["ok", "fail", "ok2"], fail_fast=False)

    assert len(results) == 3
    assert results[0].label == SentimentLabel.NEGATIVE
    assert results[1].label == SentimentLabel.NEUTRAL
    assert results[1].confidence == 0.33
    assert results[1].latency_ms == 0.0
    assert "error" in results[1].metadata
    assert results[2].label == SentimentLabel.NEGATIVE


def test_onnx_classifier_classify_batch_fail_fast():
    backend = MagicMock(spec=ONNXBackend)
    backend.model_name = "minilm-int8"
    backend.predict.side_effect = RuntimeError("Fatal error")

    clf = ONNXClassifier(backend=backend)
    with pytest.raises(ClassifierProviderError):
        clf.classify_batch(["fail"], fail_fast=True)


def test_onnx_classifier_default_live_backend():
    clf = ONNXClassifier()
    assert clf.tier == Tier.TIER_4
    assert clf.model_name == "setfit-minilm-int8"
    assert clf.backend.is_available() is True
    res = clf.classify("The product was fantastic, loved it!")
    assert isinstance(res, SentimentResult)
    assert res.label in (SentimentLabel.POSITIVE, SentimentLabel.NEUTRAL, SentimentLabel.NEGATIVE)
    batch = clf.classify_batch(["Good product", "Terrible support"])
    assert len(batch) == 2


def test_onnx_classifier_explicit_mock_backend():
    backend = MockONNXBackend(model_name="mock-onnx-minilm")
    clf = ONNXClassifier(backend=backend)
    assert clf.tier == Tier.TIER_4
    assert clf.model_name == "mock-onnx-minilm"
    assert clf.backend.is_available() is True
    res = clf.classify("Neutral review")
    assert isinstance(res, SentimentResult)
    batch = clf.classify_batch(["Good product", "Terrible support"])
    assert len(batch) == 2


def test_onnx_classifier_missing_model_path_raises():
    with pytest.raises(ClassifierProviderError, match="ONNX model artifact not found"):
        ONNXClassifier(model_path="path/does/not/exist")



def test_legacy_callable_backend_coverage():
    def dummy_fn(text: str) -> dict:
        return {"label": "NEUTRAL", "confidence": 0.5, "extra": "info"}

    clf = ONNXClassifier(inference_fn=dummy_fn, model_name="legacy-test")
    assert clf.backend.is_available() is True
    assert clf.model_name == "legacy-test"
    res = clf.classify("Hello")
    assert res.metadata == {"extra": "info"}
    batch = clf.backend.predict_batch(["one", "two"])
    assert len(batch) == 2

    # Error handling in legacy callable
    def failing_provider_fn(text: str) -> dict:
        raise ClassifierProviderError("Legacy provider error")

    clf_err = ONNXClassifier(inference_fn=failing_provider_fn)
    with pytest.raises(ClassifierProviderError, match="Legacy provider error"):
        clf_err.classify("Hello")

    def generic_failing_fn(text: str) -> dict:
        raise ValueError("Generic failure")

    clf_generic = ONNXClassifier(inference_fn=generic_failing_fn)
    with pytest.raises(ClassifierProviderError, match="Provider request failed: Generic failure"):
        clf_generic.classify("Hello")


