import json
import pytest
from pathlib import Path

from sentiments.core.models import SentimentLabel
from sentiments.providers.base import ClassifierProviderError
from sentiments.providers.tier4.packaging import ModelArtifactBundle, load_artifact_bundle


def test_load_artifact_bundle_valid(tmp_path: Path):
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"dummy onnx bytes")

    tok_file = tmp_path / "tokenizer.json"
    tok_file.write_text('{"version": "1.0"}', encoding="utf-8")

    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "model_name": "test-minilm",
                "id2label": {"0": "NEGATIVE", "1": "NEUTRAL", "2": "POSITIVE"},
                "max_length": 128,
            }
        ),
        encoding="utf-8",
    )

    bundle = load_artifact_bundle(tmp_path)
    assert isinstance(bundle, ModelArtifactBundle)
    assert bundle.model_path == model_file
    assert bundle.tokenizer_path == tok_file
    assert bundle.model_name == "test-minilm"
    assert bundle.max_length == 128
    assert bundle.id2label[0] == SentimentLabel.NEGATIVE
    assert bundle.id2label[1] == SentimentLabel.NEUTRAL
    assert bundle.id2label[2] == SentimentLabel.POSITIVE


def test_load_artifact_bundle_missing_dir(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(ClassifierProviderError, match="Model directory not found"):
        load_artifact_bundle(non_existent)


def test_load_artifact_bundle_missing_model_file(tmp_path: Path):
    tok_file = tmp_path / "tokenizer.json"
    tok_file.write_text("{}", encoding="utf-8")

    with pytest.raises(ClassifierProviderError, match="Missing ONNX model file"):
        load_artifact_bundle(tmp_path)


def test_load_artifact_bundle_missing_tokenizer(tmp_path: Path):
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"dummy")

    with pytest.raises(ClassifierProviderError, match="Missing tokenizer file"):
        load_artifact_bundle(tmp_path)


def test_load_artifact_bundle_defaults(tmp_path: Path):
    # Tests that when config.json is absent, sensible defaults apply
    model_file = tmp_path / "model_quantized.onnx"
    model_file.write_bytes(b"dummy")
    tok_file = tmp_path / "tokenizer.json"
    tok_file.write_text("{}", encoding="utf-8")

    bundle = load_artifact_bundle(tmp_path)
    assert bundle.model_path == model_file
    assert bundle.model_name == tmp_path.name
    assert bundle.max_length == 128
    # Default label mapping
    assert bundle.id2label[0] == SentimentLabel.NEGATIVE
    assert bundle.id2label[1] == SentimentLabel.NEUTRAL
    assert bundle.id2label[2] == SentimentLabel.POSITIVE


def test_load_artifact_bundle_corrupt_config(tmp_path: Path):
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"dummy")
    tok_file = tmp_path / "tokenizer.json"
    tok_file.write_text("{}", encoding="utf-8")
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ClassifierProviderError, match="Failed to parse config.json"):
        load_artifact_bundle(tmp_path)

