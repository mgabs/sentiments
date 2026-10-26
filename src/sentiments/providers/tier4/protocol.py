from typing import Any, Protocol, Sequence, runtime_checkable
from sentiments.core.models import SentimentLabel


@runtime_checkable
class ONNXBackend(Protocol):
    """Protocol defining the interface for Tier 4 ONNX inference engines."""

    @property
    def model_name(self) -> str:
        """Name or identifier of the active ONNX model."""
        ...

    def is_available(self) -> bool:
        """Check if runtime dependencies and model artifacts are available."""
        ...

    def predict(self, text: str) -> tuple[SentimentLabel, float, dict[str, Any]]:
        """Run inference on a single text string.

        Returns:
            Tuple of (SentimentLabel, confidence, metadata_dict).
        """
        ...

    def predict_batch(
        self, texts: Sequence[str]
    ) -> list[tuple[SentimentLabel, float, dict[str, Any]]]:
        """Run inference on a batch of text strings.

        Returns:
            List of (SentimentLabel, confidence, metadata_dict) tuples.
        """
        ...
