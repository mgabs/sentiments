from unittest.mock import MagicMock
import pytest

from sentiments.core.classifier import SentimentClassifier
from sentiments.core.models import Tier, SentimentLabel
from sentiments.providers.base import ClassifierProviderError
from sentiments.providers.tier2.classifier import Tier2SLMClassifier
from sentiments.providers.tier2.ollama import OllamaBackend


def test_tier2_classifier_protocol_conformance():
    backend = MagicMock()
    backend.model_name = "llama3.2:3b"
    clf = Tier2SLMClassifier(backend=backend)

    assert isinstance(clf, SentimentClassifier)
    assert clf.tier == Tier.TIER_2
    assert clf.model_name == "llama3.2:3b"


def test_tier2_classifier_default_backend():
    clf = Tier2SLMClassifier()
    assert isinstance(clf.backend, OllamaBackend)
    assert clf.model_name == "llama3.2:3b"


def test_tier2_classifier_classify_success():
    backend = MagicMock()
    backend.model_name = "qwen2.5:3b"
    backend.generate.return_value = '{"label": "POSITIVE", "confidence": 0.95, "explanation": "Excellent"}'

    clf = Tier2SLMClassifier(backend=backend)
    result = clf.classify("Fast delivery and pristine packaging.")

    assert result.label == SentimentLabel.POSITIVE
    assert result.confidence == 0.95
    assert result.tier == Tier.TIER_2
    assert result.model_name == "qwen2.5:3b"
    assert result.latency_ms > 0.0
    assert result.metadata == {"explanation": "Excellent"}
    backend.generate.assert_called_once()


def test_tier2_classifier_classify_batch():
    backend = MagicMock()
    backend.model_name = "llama3.2:3b"
    backend.generate.side_effect = [
        '{"label": "POSITIVE", "confidence": 0.9}',
        '{"label": "NEGATIVE", "confidence": 0.8}',
    ]

    clf = Tier2SLMClassifier(backend=backend)
    results = clf.classify_batch(["Great!", "Awful!"])

    assert len(results) == 2
    assert results[0].label == SentimentLabel.POSITIVE
    assert results[1].label == SentimentLabel.NEGATIVE


@pytest.mark.parametrize("invalid_input", ["", "   ", "\n\t", 123, None, []])
def test_tier2_classifier_invalid_text_raises_value_error(invalid_input):
    backend = MagicMock()
    clf = Tier2SLMClassifier(backend=backend)

    with pytest.raises(ValueError, match="text must be a non-empty string"):
        clf.classify(invalid_input)

    assert backend.generate.call_count == 0


def test_tier2_classifier_backend_exception_propagates_as_provider_error():
    backend = MagicMock()
    backend.generate.side_effect = RuntimeError("Socket dropped")

    clf = Tier2SLMClassifier(backend=backend)
    with pytest.raises(ClassifierProviderError, match="SLM inference failed: Socket dropped"):
        clf.classify("Valid review text")
