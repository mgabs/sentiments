"""Unit tests for Tier 4 SetFit training & INT8 quantization export pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from tokenizers import Tokenizer
from tokenizers.models import BPE

from sentiments.core.models import SentimentLabel
from sentiments.providers.tier4.packaging import ModelArtifactBundle, load_artifact_bundle
from sentiments.training.setfit_pipeline import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_DIR,
    LABEL_TO_ID,
    SetFitONNXWrapper,
    SetFitTrainingConfig,
    export_onnx_bundle,
    extract_linear_head,
    prepare_dataset_splits,
    resolve_dataset_path,
    run_setfit_training_pipeline,
    save_tokenizer,
    train_setfit_model,
)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.train_tier4_setfit import parse_args, main as script_main


def test_setfit_training_config_defaults():
    cfg = SetFitTrainingConfig()
    assert cfg.model_id == DEFAULT_MODEL_ID
    assert cfg.max_length == DEFAULT_MAX_LENGTH
    assert cfg.output_dir == DEFAULT_OUTPUT_DIR
    assert cfg.model_name == "setfit-minilm-int8"
    assert cfg.batch_size == 16
    assert cfg.num_epochs == 1
    assert cfg.num_iterations == 20
    assert cfg.quantize is True
    assert cfg.seed == 42


def test_resolve_dataset_path(tmp_path: Path):
    test_dir = tmp_path / "nlp-datasets" / "test"
    test_dir.mkdir(parents=True)
    (test_dir / "test.csv").write_text("Content,Tag\nNice,Positive\n", encoding="utf-8")

    resolved = resolve_dataset_path(test_dir)
    assert resolved == test_dir


def test_prepare_dataset_splits(tmp_path: Path):
    # Create sample CSV dataset
    csv_file = tmp_path / "Product (Cat) - Reviews.csv"
    lines = ["Content,Tag\n"]
    for i in range(7):
        lines.append(f"Positive review {i},Positive\n")
    for i in range(2):
        lines.append(f"Negative review {i},Negative\n")
    lines.append("Neutral review 0,Neutral\n")
    csv_file.write_text("".join(lines), encoding="utf-8")

    train_ds, val_ds, test_records = prepare_dataset_splits(
        dataset_path=tmp_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    assert len(train_ds) + len(val_ds) + len(test_records) == 10
    assert "text" in train_ds.column_names
    assert "label" in train_ds.column_names
    assert all(isinstance(l, int) for l in train_ds["label"])


def test_prepare_dataset_splits_empty_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="No records found"):
        prepare_dataset_splits(tmp_path / "empty")


def test_extract_linear_head_nn_linear():
    lin = nn.Linear(384, 3)
    extracted = extract_linear_head(lin, embedding_dim=384)
    assert extracted is lin


def test_extract_linear_head_submodule():
    class DummyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(384, 3)

    head = DummyHead()
    extracted = extract_linear_head(head, embedding_dim=384)
    assert extracted is head.linear


def test_extract_linear_head_from_sklearn_logistic_regression():
    X = np.random.randn(30, 32).astype(np.float32)
    y = np.random.choice([0, 1, 2], size=30)
    clf = LogisticRegression()
    clf.fit(X, y)

    linear = extract_linear_head(clf, embedding_dim=32, id2label={0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"})
    assert isinstance(linear, nn.Linear)
    assert linear.in_features == 32
    assert linear.out_features == 3

    # Check forward pass matches sklearn decision function
    with torch.no_grad():
        out = linear(torch.from_numpy(X)).numpy()
    sklearn_out = clf.decision_function(X)
    np.testing.assert_allclose(out, sklearn_out, rtol=1e-5, atol=1e-5)


def test_extract_linear_head_unsupported():
    with pytest.raises(TypeError, match="Unsupported head type"):
        extract_linear_head("not_a_head", embedding_dim=32)


def test_setfit_onnx_wrapper_forward():
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(50, 16)

        def forward(self, features):
            x = features["input_ids"]
            features["sentence_embedding"] = self.emb(x).mean(dim=1)
            return features

    backbone = DummyBackbone()
    head = nn.Linear(16, 3)
    wrapper = SetFitONNXWrapper(backbone, head)

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long)

    logits = wrapper(input_ids, attention_mask)
    assert logits.shape == (2, 3)


def test_save_tokenizer_fast(tmp_path: Path):
    tok = Tokenizer(BPE())
    path = save_tokenizer(tok, tmp_path)
    assert path == tmp_path / "tokenizer.json"
    assert path.is_file()


def test_save_tokenizer_failure(tmp_path: Path):
    dummy = object()
    with pytest.raises(RuntimeError, match="Failed to generate tokenizer.json"):
        save_tokenizer(dummy, tmp_path)


class TinyModel(nn.Module):
    """Tiny mock model for export tests."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 16)
        self.head = nn.Linear(16, 3)

    def forward(self, features):
        x = features["input_ids"]
        features["sentence_embedding"] = self.emb(x).mean(dim=1)
        return features


def test_export_onnx_bundle_quantized(tmp_path: Path):
    model = TinyModel()
    tok = Tokenizer(BPE())

    bundle = export_onnx_bundle(
        model=model,
        output_dir=tmp_path,
        tokenizer=tok,
        model_name="test-tiny-quantized",
        max_length=64,
        quantize=True,
    )

    assert isinstance(bundle, ModelArtifactBundle)
    assert bundle.model_name == "test-tiny-quantized"
    assert bundle.max_length == 64
    assert bundle.id2label[0] == SentimentLabel.NEGATIVE
    assert bundle.id2label[1] == SentimentLabel.NEUTRAL
    assert bundle.id2label[2] == SentimentLabel.POSITIVE
    assert (tmp_path / "model.onnx").is_file()
    assert (tmp_path / "model_quantized.onnx").is_file()
    assert (tmp_path / "tokenizer.json").is_file()
    assert (tmp_path / "config.json").is_file()

    # Re-verify via load_artifact_bundle
    reloaded = load_artifact_bundle(tmp_path)
    assert reloaded.model_path == tmp_path / "model_quantized.onnx"
    assert reloaded.tokenizer_path == tmp_path / "tokenizer.json"


def test_export_onnx_bundle_no_quantize(tmp_path: Path):
    model = TinyModel()
    tok = Tokenizer(BPE())

    bundle = export_onnx_bundle(
        model=model,
        output_dir=tmp_path,
        tokenizer=tok,
        model_name="test-tiny-fp32",
        quantize=False,
    )

    assert isinstance(bundle, ModelArtifactBundle)
    assert (tmp_path / "model.onnx").is_file()
    assert not (tmp_path / "model_quantized.onnx").exists()

    reloaded = load_artifact_bundle(tmp_path)
    assert reloaded.model_path == tmp_path / "model.onnx"


def test_export_onnx_bundle_unsupported_model(tmp_path: Path):
    with pytest.raises(TypeError, match="Unsupported model type for export"):
        export_onnx_bundle(
            model="string_not_model",
            output_dir=tmp_path,
        )


def test_train_setfit_model_mocked():
    with patch("setfit.SetFitModel.from_pretrained") as mock_from_pretrained, patch(
        "setfit.Trainer"
    ) as mock_trainer_cls:
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        cfg = SetFitTrainingConfig(batch_size=8, num_epochs=2, num_iterations=5)
        train_ds = MagicMock()
        val_ds = MagicMock()

        result = train_setfit_model(config=cfg, train_dataset=train_ds, eval_dataset=val_ds)

        assert result is mock_model
        mock_from_pretrained.assert_called_once_with(
            cfg.model_id,
            labels=["NEGATIVE", "NEUTRAL", "POSITIVE"],
        )
        mock_trainer.train.assert_called_once()


def test_run_setfit_training_pipeline_mocked(tmp_path: Path):
    with patch("sentiments.training.setfit_pipeline.prepare_dataset_splits") as mock_prep, patch(
        "sentiments.training.setfit_pipeline.train_setfit_model"
    ) as mock_train, patch(
        "sentiments.training.setfit_pipeline.export_onnx_bundle"
    ) as mock_export:
        mock_prep.return_value = (MagicMock(), MagicMock(), [])
        mock_model = MagicMock()
        mock_train.return_value = mock_model
        expected_bundle = MagicMock(spec=ModelArtifactBundle)
        mock_export.return_value = expected_bundle

        cfg = SetFitTrainingConfig(output_dir=tmp_path)
        bundle = run_setfit_training_pipeline(cfg)

        assert bundle is expected_bundle
        mock_prep.assert_called_once()
        mock_train.assert_called_once()
        mock_export.assert_called_once()


def test_train_tier4_script_cli_args():
    args = parse_args(
        [
            "--model-id",
            "test/model",
            "--dataset-path",
            "test/data",
            "--output-dir",
            "test/out",
            "--model-name",
            "custom-name",
            "--batch-size",
            "32",
            "--num-epochs",
            "3",
            "--num-iterations",
            "10",
            "--max-length",
            "64",
            "--no-quantize",
        ]
    )
    assert args.model_id == "test/model"
    assert args.dataset_path == "test/data"
    assert args.output_dir == "test/out"
    assert args.model_name == "custom-name"
    assert args.batch_size == 32
    assert args.num_epochs == 3
    assert args.num_iterations == 10
    assert args.max_length == 64
    assert args.no_quantize is True


def test_train_tier4_script_main_mocked(tmp_path: Path):
    dummy_bundle = ModelArtifactBundle(
        model_path=tmp_path / "model_quantized.onnx",
        tokenizer_path=tmp_path / "tokenizer.json",
        model_name="mock-model",
        max_length=128,
    )
    dummy_bundle.model_path.write_bytes(b"dummy")
    dummy_bundle.tokenizer_path.write_text("{}", encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({"model_name": "mock-model"}), encoding="utf-8")

    with patch("scripts.train_tier4_setfit.run_setfit_training_pipeline") as mock_pipeline, patch(
        "scripts.train_tier4_setfit.load_artifact_bundle"
    ) as mock_load:
        mock_pipeline.return_value = dummy_bundle
        mock_load.return_value = dummy_bundle

        exit_code = script_main(["--output-dir", str(tmp_path)])
        assert exit_code == 0
        mock_pipeline.assert_called_once()
        mock_load.assert_called_once_with(str(tmp_path))
