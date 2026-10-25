import json
import urllib.error
from unittest.mock import patch, MagicMock
import pytest

from sentiments.providers.base import ClassifierProviderError
from sentiments.providers.tier2.protocol import SLMBackend
from sentiments.providers.tier2.ollama import OllamaBackend


def test_ollama_backend_protocol_conformance():
    backend = OllamaBackend(model_name="llama3.2:3b")
    assert isinstance(backend, SLMBackend)
    assert backend.model_name == "llama3.2:3b"


def test_ollama_backend_generate_success():
    backend = OllamaBackend(model_name="llama3.2:3b", base_url="http://localhost:11434", timeout=5.0)

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"response": '{"label": "POSITIVE", "confidence": 0.95}'}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        out = backend.generate("Classify this review", json_schema=True)

        assert out == '{"label": "POSITIVE", "confidence": 0.95}'
        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://localhost:11434/api/generate"
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "llama3.2:3b"
        assert payload["prompt"] == "Classify this review"
        assert payload["stream"] is False
        assert payload["format"] == "json"


def test_ollama_backend_generate_connection_refused_raises():
    backend = OllamaBackend(model_name="llama3.2:3b", base_url="http://localhost:11434")

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        with pytest.raises(ClassifierProviderError, match="Ollama daemon unreachable at http://localhost:11434"):
            backend.generate("prompt")


def test_ollama_backend_generate_http_404_model_not_found():
    backend = OllamaBackend(model_name="nonexistent:model", base_url="http://localhost:11434")

    http_error = urllib.error.HTTPError(
        url="http://localhost:11434/api/generate",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=MagicMock(read=lambda: b'{"error":"model not found"}')
    )
    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(ClassifierProviderError, match="Model 'nonexistent:model' not found"):
            backend.generate("prompt")


def test_ollama_backend_generate_other_http_error():
    backend = OllamaBackend(model_name="llama3.2:3b", base_url="http://localhost:11434")

    http_error = urllib.error.HTTPError(
        url="http://localhost:11434/api/generate",
        code=500,
        msg="Internal Error",
        hdrs={},
        fp=MagicMock(read=lambda: b"Server error")
    )
    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(ClassifierProviderError, match="Ollama HTTP 500 error"):
            backend.generate("prompt")


def test_ollama_backend_is_available():
    backend = OllamaBackend(model_name="llama3.2:3b", base_url="http://localhost:11434")

    # True when ping succeeds
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    with patch("urllib.request.urlopen", return_value=mock_resp):
        assert backend.is_available() is True

    # False when ping fails
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Refused")):
        assert backend.is_available() is False
