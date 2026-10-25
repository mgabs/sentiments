import sys
from unittest.mock import patch, MagicMock
import pytest

from sentiments.providers.base import ClassifierProviderError
from sentiments.providers.tier2.protocol import SLMBackend
from sentiments.providers.tier2.mlx import MLXBackend


def test_mlx_backend_protocol_conformance():
    backend = MLXBackend(model_name="mlx-community/Qwen2.5-3B-Instruct-4bit")
    assert isinstance(backend, SLMBackend)
    assert backend.model_name == "mlx-community/Qwen2.5-3B-Instruct-4bit"


def test_mlx_backend_is_available_true_on_apple_silicon_with_mlx():
    backend = MLXBackend()
    with patch("sys.platform", "darwin"), \
         patch("platform.machine", return_value="arm64"), \
         patch.dict("sys.modules", {"mlx_lm": MagicMock()}):
        assert backend.is_available() is True


def test_mlx_backend_is_available_false_on_non_darwin():
    backend = MLXBackend()
    with patch("sys.platform", "linux"):
        assert backend.is_available() is False


def test_mlx_backend_is_available_false_on_x86_darwin():
    backend = MLXBackend()
    with patch("sys.platform", "darwin"), patch("platform.machine", return_value="x86_64"):
        assert backend.is_available() is False


def test_mlx_backend_is_available_false_when_mlx_lm_import_fails():
    backend = MLXBackend()
    with patch("sys.platform", "darwin"), \
         patch("platform.machine", return_value="arm64"), \
         patch("builtins.__import__", side_effect=ImportError("No module named mlx_lm")):
        assert backend.is_available() is False


def test_mlx_backend_generate_success():
    backend = MLXBackend(model_name="mlx-community/Llama-3.2-3B-Instruct-4bit")

    mock_mlx_lm = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
    mock_mlx_lm.generate.return_value = '{"label": "NEGATIVE", "confidence": 0.90}'

    with patch.object(backend, "is_available", return_value=True), \
         patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
        out = backend.generate("Terrible quality", json_schema=True)

        assert out == '{"label": "NEGATIVE", "confidence": 0.90}'
        mock_mlx_lm.load.assert_called_once_with("mlx-community/Llama-3.2-3B-Instruct-4bit")
        mock_mlx_lm.generate.assert_called_once_with(
            mock_model,
            mock_tokenizer,
            prompt="Terrible quality",
            max_tokens=128,
            verbose=False,
        )

        # Subsequent call reuses cached model/tokenizer
        backend.generate("Second prompt")
        assert mock_mlx_lm.load.call_count == 1


def test_mlx_backend_generate_raises_when_unavailable():
    backend = MLXBackend()
    with patch.object(backend, "is_available", return_value=False):
        with pytest.raises(ClassifierProviderError, match="MLX backend requires macOS Apple Silicon"):
            backend.generate("prompt")


def test_mlx_backend_generate_runtime_failure_raises():
    backend = MLXBackend()
    mock_mlx_lm = MagicMock()
    mock_mlx_lm.load.side_effect = RuntimeError("Metal device error")

    with patch.object(backend, "is_available", return_value=True), \
         patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
        with pytest.raises(ClassifierProviderError, match="MLX generation error"):
            backend.generate("prompt")
