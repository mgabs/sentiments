import pytest
from sentiments.core import SentimentLabel, Tier, SentimentClassifier
from sentiments.providers.base import ClassifierProviderError
from sentiments.providers.tier1 import FrontierLLMClassifier
from sentiments.providers.tier2 import CompactSLMClassifier
from sentiments.providers.tier3 import FineTunedEncoderClassifier
from sentiments.providers.tier4 import ONNXClassifier


def test_frontier_llm_classifier_success():
    def mock_client(prompt: str) -> dict:
        assert prompt == "Superb service"
        return {"label": "POSITIVE", "confidence": 0.95, "tokens": 42}

    clf = FrontierLLMClassifier(client_fn=mock_client, model_name="claude-3-5-sonnet")
    assert isinstance(clf, SentimentClassifier)
    assert clf.tier == Tier.TIER_1
    assert clf.model_name == "claude-3-5-sonnet"

    res = clf.classify("Superb service")
    assert res.label == SentimentLabel.POSITIVE
    assert res.confidence == pytest.approx(0.95)
    assert res.tier == Tier.TIER_1
    assert res.metadata["tokens"] == 42
    assert res.latency_ms >= 0.0


def test_frontier_llm_classifier_timeout():
    def mock_timeout_client(prompt: str) -> dict:
        assert prompt == "Any text"
        raise TimeoutError("Connection timed out")

    clf = FrontierLLMClassifier(client_fn=mock_timeout_client, model_name="gpt-4o")
    with pytest.raises(ClassifierProviderError, match=r"^Provider request failed: Connection timed out$"):
        clf.classify("Any text")


def test_frontier_llm_classifier_non_string():
    clf = FrontierLLMClassifier(client_fn=lambda p: {}, model_name="m")
    with pytest.raises(ValueError, match=r"^text must be a string$"):
        clf.classify(None)


def test_frontier_llm_classifier_batch():
    def mock_client(prompt: str) -> dict:
        assert prompt in ("text1", "text2")
        return {"label": "POSITIVE", "confidence": 0.90, "tokens": 10}

    clf = FrontierLLMClassifier(client_fn=mock_client, model_name="gpt-4o")
    results = clf.classify_batch(["text1", "text2"])
    assert len(results) == 2
    assert all(r.label == SentimentLabel.POSITIVE for r in results)


def test_compact_slm_classifier_success():
    def mock_slm(prompt: str) -> dict:
        assert prompt == "Awful quality"
        return {"label": "NEGATIVE", "confidence": 0.85, "tokens": 15}

    clf = CompactSLMClassifier(client_fn=mock_slm, model_name="llama-3-8b-instruct")
    assert isinstance(clf, SentimentClassifier)
    assert clf.tier == Tier.TIER_2
    assert clf.model_name == "llama-3-8b-instruct"

    res = clf.classify("Awful quality")
    assert res.label == SentimentLabel.NEGATIVE
    assert res.confidence == pytest.approx(0.85)
    assert res.tier == Tier.TIER_2
    assert res.metadata["tokens"] == 15


def test_compact_slm_classifier_error():
    def mock_failing_slm(prompt: str) -> dict:
        assert prompt == "test"
        raise ConnectionError("vLLM server down")

    clf = CompactSLMClassifier(client_fn=mock_failing_slm, model_name="llama-3-8b")
    with pytest.raises(ClassifierProviderError, match=r"^Provider request failed: vLLM server down$"):
        clf.classify("test")


def test_compact_slm_classifier_non_string():
    clf = CompactSLMClassifier(client_fn=lambda p: {}, model_name="m")
    with pytest.raises(ValueError, match=r"^text must be a string$"):
        clf.classify(123)


def test_compact_slm_classifier_batch():
    clf = CompactSLMClassifier(client_fn=lambda p: {"label": "NEUTRAL", "confidence": 0.5}, model_name="m")
    batch = clf.classify_batch(["t1", "t2"])
    assert len(batch) == 2
    assert all(r.tier == Tier.TIER_2 for r in batch)


def test_fine_tuned_encoder_classifier_success():
    def mock_encoder(text: str) -> dict:
        assert text == "It is ok"
        return {"label": "NEUTRAL", "confidence": 0.70}

    clf = FineTunedEncoderClassifier(pipeline_fn=mock_encoder, model_name="roberta-sentiment")
    assert isinstance(clf, SentimentClassifier)
    assert clf.tier == Tier.TIER_3
    assert clf.model_name == "roberta-sentiment"

    res = clf.classify("It is ok")
    assert res.label == SentimentLabel.NEUTRAL
    assert res.confidence == pytest.approx(0.70)
    assert res.tier == Tier.TIER_3


def test_fine_tuned_encoder_classifier_error():
    def mock_encoder_error(text: str) -> dict:
        assert text == "test"
        raise RuntimeError("Inference tensor error")

    clf = FineTunedEncoderClassifier(pipeline_fn=mock_encoder_error, model_name="m")
    with pytest.raises(ClassifierProviderError, match=r"^Provider request failed: Inference tensor error$"):
        clf.classify("test")


def test_fine_tuned_encoder_classifier_non_string():
    clf = FineTunedEncoderClassifier(pipeline_fn=lambda t: {}, model_name="m")
    with pytest.raises(ValueError, match=r"^text must be a string$"):
        clf.classify(None)


def test_fine_tuned_encoder_classifier_batch():
    clf = FineTunedEncoderClassifier(pipeline_fn=lambda t: {"label": "POSITIVE", "confidence": 0.9}, model_name="m")
    batch = clf.classify_batch(["t1", "t2"])
    assert len(batch) == 2
    assert all(r.tier == Tier.TIER_3 for r in batch)


def test_onnx_classifier_success():
    def mock_onnx(text: str) -> dict:
        assert text == "Very fast"
        return {"label": "POSITIVE", "confidence": 0.92}

    clf = ONNXClassifier(inference_fn=mock_onnx, model_name="quantized-distilbert.onnx")
    assert isinstance(clf, SentimentClassifier)
    assert clf.tier == Tier.TIER_4
    assert clf.model_name == "quantized-distilbert.onnx"

    res = clf.classify("Very fast")
    assert res.label == SentimentLabel.POSITIVE
    assert res.confidence == pytest.approx(0.92)
    assert res.tier == Tier.TIER_4


def test_onnx_classifier_error():
    def mock_onnx_error(text: str) -> dict:
        assert text == "test"
        raise OSError("Invalid ONNX model file")

    clf = ONNXClassifier(inference_fn=mock_onnx_error, model_name="m")
    with pytest.raises(ClassifierProviderError, match=r"^Provider request failed: Invalid ONNX model file$"):
        clf.classify("test")


def test_onnx_classifier_non_string():
    clf = ONNXClassifier(inference_fn=lambda t: {}, model_name="m")
    with pytest.raises(ValueError, match=r"^text must be a string$"):
        clf.classify(None)


def test_onnx_classifier_batch():
    clf = ONNXClassifier(inference_fn=lambda t: {"label": "NEGATIVE", "confidence": 0.8}, model_name="m")
    batch = clf.classify_batch(["t1", "t2"])
    assert len(batch) == 2
    assert all(r.tier == Tier.TIER_4 for r in batch)


def test_frontier_llm_classifier_no_tokens_metadata():
    clf = FrontierLLMClassifier(client_fn=lambda p: {"label": "POSITIVE", "confidence": 0.8}, model_name="m")
    res = clf.classify("sample")
    assert "tokens" not in res.metadata


def test_compact_slm_classifier_no_tokens_metadata():
    clf = CompactSLMClassifier(client_fn=lambda p: {"label": "NEGATIVE", "confidence": 0.8}, model_name="m")
    res = clf.classify("sample")
    assert "tokens" not in res.metadata


def test_providers_latency_calculation(monkeypatch):
    import time
    times = [10.0, 10.005]  # delta = 0.005s = 5.0ms
    monkeypatch.setattr(time, "perf_counter", lambda: times.pop(0))
    t1 = FrontierLLMClassifier(client_fn=lambda p: {"label": "POSITIVE", "confidence": 0.9}, model_name="m")
    res1 = t1.classify("hello")
    assert res1.latency_ms == pytest.approx(5.0)

    times = [10.0, 10.005]
    t2 = CompactSLMClassifier(client_fn=lambda p: {"label": "POSITIVE", "confidence": 0.9}, model_name="m")
    res2 = t2.classify("hello")
    assert res2.latency_ms == pytest.approx(5.0)

    times = [10.0, 10.005]
    t3 = FineTunedEncoderClassifier(pipeline_fn=lambda p: {"label": "POSITIVE", "confidence": 0.9}, model_name="m")
    res3 = t3.classify("hello")
    assert res3.latency_ms == pytest.approx(5.0)

    times = [10.0, 10.005]
    t4 = ONNXClassifier(inference_fn=lambda p: {"label": "POSITIVE", "confidence": 0.9}, model_name="m")
    res4 = t4.classify("hello")
    assert res4.latency_ms == pytest.approx(5.0)

