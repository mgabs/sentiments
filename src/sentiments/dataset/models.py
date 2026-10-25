from dataclasses import dataclass, field
from typing import Any

from sentiments.core import SentimentLabel


@dataclass(frozen=True)
class DatasetRecord:
    id: str
    text: str
    label: SentimentLabel
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("text cannot be empty")
        if not self.id:
            raise ValueError("id cannot be empty")


@dataclass(frozen=True)
class DatasetSplit:
    train: list[DatasetRecord]
    val: list[DatasetRecord]
    test: list[DatasetRecord]
