import time
from pathlib import Path
from typing import Any, Callable, Sequence

from sentiments.core.classifier import SentimentClassifier
from sentiments.core.models import SentimentLabel, SentimentResult, Tier
from sentiments.providers.base import ClassifierProviderError
from sentiments.providers.tier4.mock import MockONNXBackend
from sentiments.providers.tier4.packaging import ModelArtifactBundle
from sentiments.providers.tier4.protocol import ONNXBackend


class ONNXClassifier:
    """Tier 4 ultra-lightweight ONNX CPU sentiment classifier."""

    def __init__(
        self,
        backend: ONNXBackend | None = None,
        model_path: str | Path | ModelArtifactBundle | None = None,
        inference_fn: Callable[[str], dict[str, Any]] | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize ONNXClassifier.

        Accepts:
          - A pre-configured `backend: ONNXBackend`
          - Or a `model_path: str | Path | ModelArtifactBundle`
          - Or a legacy `inference_fn` for backwards compatibility
        """
        if backend is not None:
            self.backend: ONNXBackend = backend
        elif inference_fn is not None:
            # Backwards-compatibility wrapper for legacy callable
            name = model_name or "legacy-onnx"
            self.backend = _LegacyCallableBackend(inference_fn, name)
        else:
            from sentiments.providers.tier4.engine import ONNXRuntimeBackend

            default_dir = Path("models/tier4")
            target = model_path if model_path is not None else default_dir
            if isinstance(target, (str, Path)) and not Path(target).exists():
                raise ClassifierProviderError(
                    f"ONNX model artifact not found at '{target}'. Provide a valid model_path or backend."
                )
            self.backend = ONNXRuntimeBackend(target)

    @property
    def tier(self) -> Tier:
        return Tier.TIER_4

    @property
    def model_name(self) -> str:
        return self.backend.model_name

    def classify(self, text: str) -> SentimentResult:
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if not text.strip():
            raise ValueError("text must be a non-empty string")

        t0 = time.perf_counter()
        try:
            label, confidence, metadata = self.backend.predict(text)
        except ClassifierProviderError:
            raise
        except Exception as exc:
            raise ClassifierProviderError(f"ONNX inference failed: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return SentimentResult(
            label=label,
            confidence=confidence,
            latency_ms=latency_ms,
            tier=self.tier,
            model_name=self.model_name,
            metadata=metadata,
        )

    def classify_batch(
        self, texts: Sequence[str], fail_fast: bool = False
    ) -> list[SentimentResult]:
        results = []
        for t in texts:
            try:
                results.append(self.classify(t))
            except Exception as exc:
                if fail_fast:
                    raise
                results.append(
                    SentimentResult(
                        label=SentimentLabel.NEUTRAL,
                        confidence=0.33,
                        latency_ms=0.0,
                        tier=self.tier,
                        model_name=self.model_name,
                        metadata={"error": str(exc)},
                    )
                )
        return results


class _LegacyCallableBackend:
    """Adapter for legacy callable inference_fn."""

    def __init__(self, fn: Callable[[str], dict[str, Any]], model_name: str) -> None:
        self.fn = fn
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        return True

    def predict(self, text: str) -> tuple[SentimentLabel, float, dict[str, Any]]:
        try:
            resp = self.fn(text)
        except ClassifierProviderError:
            raise
        except Exception as exc:
            raise ClassifierProviderError(f"Provider request failed: {exc}") from exc
        label = SentimentLabel(resp["label"])
        confidence = float(resp["confidence"])
        meta = {k: v for k, v in resp.items() if k not in ("label", "confidence")}
        return label, confidence, meta

    def predict_batch(
        self, texts: Sequence[str]
    ) -> list[tuple[SentimentLabel, float, dict[str, Any]]]:
        return [self.predict(t) for t in texts]
