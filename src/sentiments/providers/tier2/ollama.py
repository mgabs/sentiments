import json
import urllib.error
import urllib.request
from typing import Any
from sentiments.providers.base import ClassifierProviderError


class OllamaBackend:
    """Backend adapter communicating with local Ollama HTTP daemon via standard library urllib."""

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
    ) -> None:
        self._model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, json_schema: bool = True) -> str:
        url = f"{self.base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
        }
        if json_schema:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                res_json = json.loads(body)
                return str(res_json.get("response", ""))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise ClassifierProviderError(
                    f"Model '{self._model_name}' not found on Ollama at {self.base_url}. Run 'ollama pull {self._model_name}'."
                ) from exc
            raise ClassifierProviderError(
                f"Ollama HTTP {exc.code} error: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ClassifierProviderError(
                f"Ollama daemon unreachable at {self.base_url}. Run 'ollama serve'."
            ) from exc

    def is_available(self) -> bool:
        url = f"{self.base_url}/api/version"
        req = urllib.request.Request(url=url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=min(2.0, self.timeout)):
                return True
        except Exception:
            return False
