import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from onnxruntime.datasets import get_example
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordPiece

from sentiments.core.models import SentimentLabel, Tier
from sentiments.providers.base import ClassifierProviderError
from sentiments.providers.tier4.classifier import ONNXClassifier
from sentiments.providers.tier4.engine import ONNXRuntimeBackend, stable_softmax
from sentiments.providers.tier4.packaging import ModelArtifactBundle, load_artifact_bundle


@pytest.fixture
def sample_bundle_dir(tmp_path: Path) -> Path:
    """Create a temporary valid ONNX bundle directory with model and tokenizer."""
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "great": 2,
        "terrible": 3,
        "service": 4,
        "ok": 5,
        "good": 6,
        "bad": 7,
    }
    tok = Tokenizer(WordPiece(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok_path = tmp_path / "tokenizer.json"
    tok.save(str(tok_path))

    model_src = get_example("mul_1.onnx")
    model_dst = tmp_path / "model.onnx"
    shutil.copyfile(model_src, model_dst)

    config = {
        "model_name": "test-onnx-suite",
        "max_length": 6,
        "id2label": {"0": "NEGATIVE", "1": "NEUTRAL", "2": "POSITIVE"},
        "metadata": {"unit_test": True},
    }
    with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f)

    return tmp_path


def test_stable_softmax_basic():
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    probs = stable_softmax(logits)
    assert probs.shape == (3,)
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[2] > probs[1] > probs[0]


def test_stable_softmax_numerical_stability():
    # Large positive logits: would overflow standard exp()
    huge_pos = np.array([[10000.0, 10005.0, 9995.0]])
    p_pos = stable_softmax(huge_pos)
    assert not np.isnan(p_pos).any()
    assert not np.isinf(p_pos).any()
    assert np.isclose(np.sum(p_pos), 1.0)
    assert np.argmax(p_pos) == 1

    # Large negative logits
    huge_neg = np.array([[-10000.0, -10005.0, -9995.0]])
    p_neg = stable_softmax(huge_neg)
    assert not np.isnan(p_neg).any()
    assert not np.isinf(p_neg).any()
    assert np.isclose(np.sum(p_neg), 1.0)
    assert np.argmax(p_neg) == 2


def test_stable_softmax_batch_2d():
    logits = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    probs = stable_softmax(logits, axis=-1)
    assert probs.shape == (3, 3)
    assert np.allclose(np.sum(probs, axis=-1), 1.0)
    assert np.argmax(probs[0]) == 0
    assert np.argmax(probs[1]) == 1
    assert np.argmax(probs[2]) == 2


def test_init_with_valid_bundle_dir(sample_bundle_dir: Path):
    backend = ONNXRuntimeBackend(sample_bundle_dir)
    assert backend.is_available() is True
    assert backend.model_name == "test-onnx-suite"
    assert backend.bundle.max_length == 6


def test_init_with_model_artifact_bundle(sample_bundle_dir: Path):
    bundle = load_artifact_bundle(sample_bundle_dir)
    backend = ONNXRuntimeBackend(bundle)
    assert backend.is_available() is True
    assert backend.model_name == "test-onnx-suite"


def test_init_invalid_bundle_type():
    with pytest.raises(ClassifierProviderError, match="Invalid bundle or model path"):
        ONNXRuntimeBackend(12345)  # type: ignore


def test_init_missing_dependencies_raises_provider_error(sample_bundle_dir: Path):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name in ("onnxruntime", "tokenizers"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(ClassifierProviderError, match="onnxruntime and tokenizers are required"):
            ONNXRuntimeBackend(sample_bundle_dir)


def test_init_corrupted_tokenizer_raises_provider_error(tmp_path: Path):
    corrupt_tok = tmp_path / "tokenizer.json"
    corrupt_tok.write_text("invalid json content")

    model_src = get_example("mul_1.onnx")
    shutil.copyfile(model_src, tmp_path / "model.onnx")

    bundle = ModelArtifactBundle(
        model_path=tmp_path / "model.onnx",
        tokenizer_path=corrupt_tok,
        model_name="broken-tok",
    )
    with pytest.raises(ClassifierProviderError, match="Failed to load tokenizer"):
        ONNXRuntimeBackend(bundle)


def test_init_corrupted_model_raises_provider_error(tmp_path: Path, sample_bundle_dir: Path):
    corrupt_model = tmp_path / "model.onnx"
    corrupt_model.write_bytes(b"garbage binary not onnx")

    tok_path = sample_bundle_dir / "tokenizer.json"

    bundle = ModelArtifactBundle(
        model_path=corrupt_model,
        tokenizer_path=tok_path,
        model_name="broken-model",
    )
    with pytest.raises(ClassifierProviderError, match="Failed to initialize ONNX InferenceSession"):
        ONNXRuntimeBackend(bundle)


def test_predict_single_sample(sample_bundle_dir: Path):
    backend = ONNXRuntimeBackend(sample_bundle_dir)
    backend.session.run = MagicMock(
        return_value=[np.array([[0.1, 0.2, 0.7]], dtype=np.float32)]
    )

    label, conf, meta = backend.predict("great service")
    assert label == SentimentLabel.POSITIVE
    assert conf > 0.4
    assert meta == {"unit_test": True}


def test_predict_batch_samples(sample_bundle_dir: Path):
    backend = ONNXRuntimeBackend(sample_bundle_dir)
    backend.session.run = MagicMock(
        return_value=[
            np.array(
                [
                    [0.8, 0.1, 0.1],  # Row 0: NEGATIVE
                    [0.1, 0.8, 0.1],  # Row 1: NEUTRAL
                    [0.1, 0.1, 0.8],  # Row 2: POSITIVE
                ],
                dtype=np.float32,
            )
        ]
    )

    results = backend.predict_batch(["terrible bad", "ok ok", "great good"])
    assert len(results) == 3
    assert results[0][0] == SentimentLabel.NEGATIVE
    assert results[1][0] == SentimentLabel.NEUTRAL
    assert results[2][0] == SentimentLabel.POSITIVE
    assert all(r[1] > 0.5 for r in results)
    assert all(r[2] == {"unit_test": True} for r in results)


def test_predict_batch_empty(sample_bundle_dir: Path):
    backend = ONNXRuntimeBackend(sample_bundle_dir)
    assert backend.predict_batch([]) == []


def test_predict_invalid_inputs(sample_bundle_dir: Path):
    backend = ONNXRuntimeBackend(sample_bundle_dir)
    with pytest.raises(ValueError, match="text must be a string"):
        backend.predict(None)  # type: ignore
    with pytest.raises(ValueError, match="All elements of texts must be strings"):
        backend.predict_batch(["ok", 123])  # type: ignore


def test_oversized_text_truncation(sample_bundle_dir: Path):
    backend = ONNXRuntimeBackend(sample_bundle_dir)
    captured_feed = {}

    def mock_run(out_names, feed):
        captured_feed.update(feed)
        batch_size = next(iter(feed.values())).shape[0]
        return [np.zeros((batch_size, 3), dtype=np.float32)]

    backend.session.run = mock_run

    long_text = "great service " * 50  # 100 words >> max_length=6
    backend.predict(long_text)

    # First input in feed_dict must be truncated to max_length (6)
    first_tensor = next(iter(captured_feed.values()))
    assert first_tensor.shape[1] <= 6


def test_predict_session_run_failure(sample_bundle_dir: Path):
    backend = ONNXRuntimeBackend(sample_bundle_dir)
    backend.session.run = MagicMock(side_effect=RuntimeError("Device computation failed"))

    with pytest.raises(ClassifierProviderError, match="ONNX inference failed: Device computation failed"):
        backend.predict("great service")


def test_is_available_returns_false_on_missing_file(sample_bundle_dir: Path):
    backend = ONNXRuntimeBackend(sample_bundle_dir)
    assert backend.is_available() is True

    # Delete the model file
    (sample_bundle_dir / "model.onnx").unlink()
    assert backend.is_available() is False


def test_onnx_classifier_with_model_path_integration(sample_bundle_dir: Path):
    clf = ONNXClassifier(model_path=sample_bundle_dir)
    assert clf.tier == Tier.TIER_4
    assert clf.model_name == "test-onnx-suite"

    # Mock session run to return POSITIVE prediction
    clf.backend.session.run = MagicMock(
        return_value=[np.array([[0.0, 0.0, 5.0]], dtype=np.float32)]
    )

    result = clf.classify("great service")
    assert result.label == SentimentLabel.POSITIVE
    assert result.confidence > 0.8
    assert result.tier == Tier.TIER_4
    assert result.model_name == "test-onnx-suite"
    assert result.metadata == {"unit_test": True}


def test_onnx_classifier_with_bundle_object(sample_bundle_dir: Path):
    bundle = load_artifact_bundle(sample_bundle_dir)
    clf = ONNXClassifier(model_path=bundle)
    assert clf.model_name == "test-onnx-suite"


def test_feed_dict_with_multiple_inputs_mapping(sample_bundle_dir: Path):
    backend = ONNXRuntimeBackend(sample_bundle_dir)

    # Mock get_inputs to simulate bert-style 3 inputs: input_ids, attention_mask, token_type_ids
    inp_ids = MagicMock()
    inp_ids.name = "input_ids"
    inp_ids.type = "tensor(int64)"

    inp_mask = MagicMock()
    inp_mask.name = "attention_mask"
    inp_mask.type = "tensor(int64)"

    inp_type = MagicMock()
    inp_type.name = "token_type_ids"
    inp_type.type = "tensor(int32)"

    backend.session.get_inputs = MagicMock(return_value=[inp_ids, inp_mask, inp_type])

    recorded_feed = {}

    def mock_run(out_names, feed):
        recorded_feed.update(feed)
        return [np.array([[0.3, 0.4, 0.3]], dtype=np.float32)]

    backend.session.run = mock_run

    backend.predict("ok service")

    assert "input_ids" in recorded_feed
    assert "attention_mask" in recorded_feed
    assert "token_type_ids" in recorded_feed
    assert recorded_feed["input_ids"].dtype == np.int64
    assert recorded_feed["attention_mask"].dtype == np.int64
    assert recorded_feed["token_type_ids"].dtype == np.int32


def test_1d_logits_and_unknown_label_fallback(sample_bundle_dir: Path):
    bundle = load_artifact_bundle(sample_bundle_dir)
    # Set id2label without index 2 to test fallback
    restricted_bundle = ModelArtifactBundle(
        model_path=bundle.model_path,
        tokenizer_path=bundle.tokenizer_path,
        model_name="restricted",
        max_length=bundle.max_length,
        id2label={0: SentimentLabel.NEGATIVE, 1: SentimentLabel.NEUTRAL},
    )
    backend = ONNXRuntimeBackend(restricted_bundle)

    # Return 1D array where argmax is index 2 (unmapped)
    backend.session.run = MagicMock(
        return_value=[np.array([0.1, 0.2, 0.9], dtype=np.float32)]
    )

    label, conf, meta = backend.predict("test")
    assert label == SentimentLabel.NEUTRAL
    assert conf > 0.5


def test_is_available_handles_exception(sample_bundle_dir: Path):
    backend = ONNXRuntimeBackend(sample_bundle_dir)
    with patch.object(Path, "exists", side_effect=RuntimeError("disk error")):
        assert backend.is_available() is False


def test_predict_no_results_raises_provider_error(sample_bundle_dir: Path):
    backend = ONNXRuntimeBackend(sample_bundle_dir)
    backend.predict_batch = MagicMock(return_value=[])
    with pytest.raises(ClassifierProviderError, match="No prediction returned from backend"):
        backend.predict("test")


def test_predict_batch_flat_array_reshape(sample_bundle_dir: Path):
    backend = ONNXRuntimeBackend(sample_bundle_dir)
    # Return 1D array of 6 elements for 2 texts
    backend.session.run = MagicMock(
        return_value=[np.array([0.1, 0.2, 0.7, 0.8, 0.1, 0.1], dtype=np.float32)]
    )
    res = backend.predict_batch(["text1", "text2"])
    assert len(res) == 2
    assert res[0][0] == SentimentLabel.POSITIVE
    assert res[1][0] == SentimentLabel.NEGATIVE


def test_predict_batch_classifier_provider_error_reraises(sample_bundle_dir: Path):
    backend = ONNXRuntimeBackend(sample_bundle_dir)
    backend.session.run = MagicMock(side_effect=ClassifierProviderError("Custom failure"))
    with pytest.raises(ClassifierProviderError, match="Custom failure"):
        backend.predict_batch(["test"])

