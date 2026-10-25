from typing import Protocol, runtime_checkable

from sentiments.core.models import SentimentResult, Tier


@runtime_checkable
class SentimentClassifier(Protocol):
    @property
    def tier(self) -> Tier:
        """The architectural tier of this classifier."""
        ...

    @property
    def model_name(self) -> str:
        """The identifier/name of the classifier model."""
        ...

    def classify(self, text: str) -> SentimentResult:
        """Classify sentiment of a single text string."""
        ...

    def classify_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Classify sentiment of a batch of text strings."""
        ...
