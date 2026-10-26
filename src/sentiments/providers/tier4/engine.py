"""ONNX Runtime inference backend for Tier 4 CPU sentiment classification."""

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sentiments.core.models import SentimentLabel
from sentiments.providers.base import ClassifierProviderError
from sentiments.providers.tier4.packaging import ModelArtifactBundle, load_artifact_bundle


def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute numerically stable softmax along the specified axis."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)
    return exp_shifted / sum_exp


class ONNXRuntimeBackend:
    """Production ONNX Runtime execution engine using tokenizers and CPUExecutionProvider."""

    def __init__(self, bundle_or_path: ModelArtifactBundle | str | Path) -> None:
        try:
            import onnxruntime
            import tokenizers
        except (ImportError, ModuleNotFoundError) as exc:
            raise ClassifierProviderError(
                f"onnxruntime and tokenizers are required for ONNXRuntimeBackend: {exc}"
            ) from exc

        if isinstance(bundle_or_path, ModelArtifactBundle):
            self.bundle = bundle_or_path
        elif isinstance(bundle_or_path, (str, Path)):
            self.bundle = load_artifact_bundle(bundle_or_path)
        else:
            raise ClassifierProviderError(f"Invalid bundle or model path: {bundle_or_path}")

        try:
            self.tokenizer = tokenizers.Tokenizer.from_file(str(self.bundle.tokenizer_path))
            self.tokenizer.enable_truncation(max_length=self.bundle.max_length)
            self.tokenizer.enable_padding()
        except Exception as exc:
            raise ClassifierProviderError(
                f"Failed to load tokenizer from {self.bundle.tokenizer_path}: {exc}"
            ) from exc

        try:
            opts = onnxruntime.SessionOptions()
            opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = onnxruntime.InferenceSession(
                str(self.bundle.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise ClassifierProviderError(
                f"Failed to initialize ONNX InferenceSession: {exc}"
            ) from exc

    @property
    def model_name(self) -> str:
        """Name or identifier of the active ONNX model."""
        return self.bundle.model_name

    def is_available(self) -> bool:
        """Check if runtime dependencies and model artifacts are available."""
        try:
            import onnxruntime  # noqa: F401
            import tokenizers  # noqa: F401

            return (
                self.session is not None
                and self.tokenizer is not None
                and self.bundle.model_path.exists()
                and self.bundle.tokenizer_path.exists()
            )
        except Exception:
            return False

    def predict(self, text: str) -> tuple[SentimentLabel, float, dict[str, Any]]:
        """Run inference on a single text string."""
        if not isinstance(text, str):
            raise ValueError("text must be a string")

        results = self.predict_batch([text])
        if not results:
            raise ClassifierProviderError("No prediction returned from backend")
        return results[0]

    def predict_batch(
        self, texts: Sequence[str]
    ) -> list[tuple[SentimentLabel, float, dict[str, Any]]]:
        """Run vectorized batch inference on a sequence of text strings."""
        if not texts:
            return []

        for t in texts:
            if not isinstance(t, str):
                raise ValueError("All elements of texts must be strings")

        try:
            encodings = self.tokenizer.encode_batch(list(texts))

            input_ids = [enc.ids for enc in encodings]
            attention_mask = [enc.attention_mask for enc in encodings]
            type_ids = [enc.type_ids for enc in encodings]

            session_inputs = self.session.get_inputs()
            feed_dict: dict[str, np.ndarray] = {}

            for inp in session_inputs:
                name = inp.name
                inp_type = getattr(inp, "type", "")
                if "float" in inp_type:
                    dtype = np.float32
                elif "int32" in inp_type:
                    dtype = np.int32
                else:
                    dtype = np.int64

                lower_name = name.lower()

                if name == "input_ids" or (
                    "input" in lower_name and "mask" not in lower_name and "type" not in lower_name
                ):
                    feed_dict[name] = np.array(input_ids, dtype=dtype)
                elif name == "attention_mask" or "mask" in lower_name:
                    feed_dict[name] = np.array(attention_mask, dtype=dtype)
                elif name in ("token_type_ids", "segment_ids") or "type" in lower_name:
                    feed_dict[name] = np.array(type_ids, dtype=dtype)
                elif len(session_inputs) == 1:
                    feed_dict[name] = np.array(input_ids, dtype=dtype)

            outputs = self.session.run(None, feed_dict)
            logits = np.asarray(outputs[0], dtype=np.float32)

            # Reshape logits to (N, num_classes)
            if logits.ndim == 1:
                logits = np.expand_dims(logits, axis=0)
            if logits.shape[0] != len(texts):
                logits = logits.reshape(len(texts), -1)

            probs = stable_softmax(logits, axis=-1)

            results: list[tuple[SentimentLabel, float, dict[str, Any]]] = []
            for i in range(len(texts)):
                row_probs = probs[i]
                best_idx = int(np.argmax(row_probs))
                confidence = float(row_probs[best_idx])
                raw_label = self.bundle.id2label.get(best_idx, SentimentLabel.NEUTRAL)
                label = (
                    SentimentLabel(raw_label)
                    if not isinstance(raw_label, SentimentLabel)
                    else raw_label
                )
                metadata = dict(self.bundle.metadata)
                results.append((label, confidence, metadata))

            return results
        except ClassifierProviderError:
            raise
        except Exception as exc:
            raise ClassifierProviderError(f"ONNX inference failed: {exc}") from exc
