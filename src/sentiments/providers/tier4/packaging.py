import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sentiments.core.models import SentimentLabel
from sentiments.providers.base import ClassifierProviderError


@dataclass(frozen=True)
class ModelArtifactBundle:
    """Encapsulates a loaded and validated Tier 4 ONNX model bundle."""

    model_path: Path
    tokenizer_path: Path
    model_name: str
    max_length: int = 128
    id2label: dict[int, SentimentLabel] = field(
        default_factory=lambda: {
            0: SentimentLabel.NEGATIVE,
            1: SentimentLabel.NEUTRAL,
            2: SentimentLabel.POSITIVE,
        }
    )
    metadata: dict[str, Any] = field(default_factory=dict)


def load_artifact_bundle(directory_path: str | Path) -> ModelArtifactBundle:
    """Validate and load an ONNX artifact bundle from disk.

    Expects:
      - A model file: `model.onnx` or `model_quantized.onnx`
      - A tokenizer file: `tokenizer.json`
      - Optional configuration: `config.json`

    Raises:
      ClassifierProviderError: If directory or required files are missing or unreadable.
    """
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        raise ClassifierProviderError(f"Model directory not found: {path}")

    # Discover ONNX model file
    model_file: Path | None = None
    for candidate in ["model_quantized.onnx", "model.onnx"]:
        cand_path = path / candidate
        if cand_path.is_file():
            model_file = cand_path
            break

    if model_file is None:
        raise ClassifierProviderError(
            f"Missing ONNX model file in {path}. Expected 'model_quantized.onnx' or 'model.onnx'."
        )

    # Discover tokenizer file
    tok_file = path / "tokenizer.json"
    if not tok_file.is_file():
        raise ClassifierProviderError(
            f"Missing tokenizer file in {path}. Expected 'tokenizer.json'."
        )

    # Parse optional config.json
    config_file = path / "config.json"
    model_name = path.name
    max_length = 128
    id2label = {
        0: SentimentLabel.NEGATIVE,
        1: SentimentLabel.NEUTRAL,
        2: SentimentLabel.POSITIVE,
    }
    metadata: dict[str, Any] = {}

    if config_file.is_file():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            if isinstance(cfg_data, dict):
                model_name = str(cfg_data.get("model_name", model_name))
                max_length = int(cfg_data.get("max_length", max_length))
                raw_id2label = cfg_data.get("id2label")
                if isinstance(raw_id2label, dict):
                    id2label = {int(k): SentimentLabel(v) for k, v in raw_id2label.items()}
                metadata = cfg_data.get("metadata", {})
        except Exception as exc:
            raise ClassifierProviderError(f"Failed to parse config.json in {path}: {exc}") from exc

    return ModelArtifactBundle(
        model_path=model_file,
        tokenizer_path=tok_file,
        model_name=model_name,
        max_length=max_length,
        id2label=id2label,
        metadata=metadata,
    )
