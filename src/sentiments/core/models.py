from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class Tier(str, Enum):
    TIER_0 = "TIER_0"
    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"
    TIER_3 = "TIER_3"
    TIER_4 = "TIER_4"


@dataclass(frozen=True)
class SentimentResult:
    label: SentimentLabel
    confidence: float
    latency_ms: float
    tier: Tier
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.latency_ms < 0.0:
            raise ValueError(f"latency_ms must be non-negative, got {self.latency_ms}")
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
