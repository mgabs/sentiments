import time
from typing import Callable, Any

from sentiments.core import SentimentLabel, Tier, SentimentResult
from sentiments.providers.base import ClassifierProviderError
from sentiments.providers.tier2.core import build_sentiment_prompt, parse_sentiment_output
from sentiments.providers.tier2.protocol import SLMBackend
from sentiments.providers.tier2.ollama import OllamaBackend
from sentiments.providers.tier2.mlx import MLXBackend
from sentiments.providers.tier2.classifier import Tier2SLMClassifier


class CompactSLMClassifier:
    """Legacy/mock-friendly Tier 2 wrapper maintaining backwards compatibility."""

    def __init__(self, client_fn: Callable[[str], dict[str, Any]], model_name: str) -> None:
        self.client_fn = client_fn
        self._model_name = model_name

    @property
    def tier(self) -> Tier:
        return Tier.TIER_2

    @property
    def model_name(self) -> str:
        return self._model_name

    def classify(self, text: str) -> SentimentResult:
        if not isinstance(text, str):
            raise ValueError("text must be a string")

        t0 = time.perf_counter()
        try:
            resp = self.client_fn(text)
        except Exception as e:
            raise ClassifierProviderError(f"Provider request failed: {e}") from e

        latency = (time.perf_counter() - t0) * 1000.0

        label = SentimentLabel(resp["label"])
        confidence = float(resp["confidence"])
        metadata: dict[str, Any] = {}
        if "tokens" in resp:
            metadata["tokens"] = resp["tokens"]

        return SentimentResult(
            label=label,
            confidence=confidence,
            latency_ms=latency,
            tier=self.tier,
            model_name=self.model_name,
            metadata=metadata,
        )

    def classify_batch(self, texts: list[str]) -> list[SentimentResult]:
        return [self.classify(t) for t in texts]


__all__ = [
    "Tier2SLMClassifier",
    "CompactSLMClassifier",
    "SLMBackend",
    "OllamaBackend",
    "MLXBackend",
    "build_sentiment_prompt",
    "parse_sentiment_output",
]
