import importlib
import platform
import sys
from typing import Any
from sentiments.providers.base import ClassifierProviderError


class MLXBackend:
    """Backend adapter executing local SLM inference on Apple Silicon via MLX (mlx-lm)."""

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen2.5-3B-Instruct-4bit",
        max_tokens: int = 128,
    ) -> None:
        self._model_name = model_name
        self.max_tokens = max_tokens
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        if sys.platform != "darwin":
            return False
        if platform.machine() not in ("arm64", "aarch64"):
            return False
        try:
            importlib.import_module("mlx_lm")
            return True
        except Exception:
            return False

    def generate(self, prompt: str, json_schema: bool = True) -> str:
        if not self.is_available():
            raise ClassifierProviderError(
                "MLX backend requires macOS Apple Silicon (arm64) and 'mlx-lm'. Install with 'uv pip install mlx-lm'."
            )

        try:
            mlx_lm = importlib.import_module("mlx_lm")
            if self._model is None or self._tokenizer is None:
                self._model, self._tokenizer = mlx_lm.load(self._model_name)

            out = mlx_lm.generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                verbose=False,
            )
            return str(out)
        except Exception as exc:
            raise ClassifierProviderError(f"MLX generation error: {exc}") from exc
