from typing import Any, Sequence
from sentiments.core.models import SentimentLabel


class MockONNXBackend:
    """Deterministic in-memory mock backend for Tier 4 ONNX testing."""

    def __init__(self, model_name: str = "mock-onnx-minilm") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        return True

    def predict(self, text: str) -> tuple[SentimentLabel, float, dict[str, Any]]:
        lower = text.lower()
        if any(w in lower for w in ["great", "outstanding", "wonderful", "amazing", "good"]):
            return SentimentLabel.POSITIVE, 0.95, {"rule": "mock-positive"}
        if any(w in lower for w in ["terrible", "bad", "broke", "defective", "horrible"]):
            return SentimentLabel.NEGATIVE, 0.95, {"rule": "mock-negative"}
        return SentimentLabel.NEUTRAL, 0.70, {"rule": "mock-neutral"}

    def predict_batch(
        self, texts: Sequence[str]
    ) -> list[tuple[SentimentLabel, float, dict[str, Any]]]:
        return [self.predict(t) for t in texts]
