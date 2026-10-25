import time
from typing import Sequence

from sentiments.core.classifier import SentimentClassifier
from sentiments.core.models import Tier, SentimentResult
from sentiments.providers.base import ClassifierProviderError
from sentiments.providers.tier2.core import build_sentiment_prompt, parse_sentiment_output
from sentiments.providers.tier2.protocol import SLMBackend
from sentiments.providers.tier2.ollama import OllamaBackend


class Tier2SLMClassifier:
    """Tier 2 local Small Language Model (SLM) sentiment classifier.

    Supports pluggable backends:
      - OllamaBackend: Local HTTP daemon client (default).
      - MLXBackend: Native Apple Silicon Metal acceleration.
    """

    def __init__(self, backend: SLMBackend | None = None) -> None:
        self.backend = backend if backend is not None else OllamaBackend()

    @property
    def tier(self) -> Tier:
        return Tier.TIER_2

    @property
    def model_name(self) -> str:
        return self.backend.model_name

    def classify(self, text: str) -> SentimentResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        prompt = build_sentiment_prompt(text)

        t0 = time.perf_counter()
        try:
            raw_output = self.backend.generate(prompt, json_schema=True)
        except ClassifierProviderError:
            raise
        except Exception as exc:
            raise ClassifierProviderError(f"SLM inference failed: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        label, confidence, metadata = parse_sentiment_output(raw_output)

        return SentimentResult(
            label=label,
            confidence=confidence,
            latency_ms=latency_ms,
            tier=self.tier,
            model_name=self.model_name,
            metadata=metadata,
        )

    def classify_batch(self, texts: Sequence[str]) -> list[SentimentResult]:
        return [self.classify(t) for t in texts]
