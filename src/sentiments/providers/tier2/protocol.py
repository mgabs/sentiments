from typing import Protocol, runtime_checkable


@runtime_checkable
class SLMBackend(Protocol):
    """Protocol for local Small Language Model execution backends."""

    @property
    def model_name(self) -> str:
        """Model identifier (e.g. llama3.2:3b, qwen2.5:3b)."""
        ...

    def generate(self, prompt: str, json_schema: bool = True) -> str:
        """Generate raw text response from the model given prompt."""
        ...

    def is_available(self) -> bool:
        """Check if backend daemon/engine is reachable and operational."""
        ...
