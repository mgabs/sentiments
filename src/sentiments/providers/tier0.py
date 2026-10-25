import re
import time
from collections import Counter
from typing import Optional

from sentiments.core import SentimentLabel, Tier, SentimentResult
from sentiments.dataset.models import DatasetRecord


class MajorityClassClassifier:
    def __init__(self) -> None:
        self.majority_label: Optional[SentimentLabel] = None
        self.confidence: float = 0.0

    @property
    def tier(self) -> Tier:
        return Tier.TIER_0

    @property
    def model_name(self) -> str:
        return "majority_class_baseline"

    def fit(self, records: list[DatasetRecord]) -> None:
        if not records:
            raise ValueError("cannot fit on empty dataset")

        counts = Counter(r.label for r in records)
        max_count = max(counts.values())

        # Deterministic tie breaking: alphabetical order of label value
        candidates = sorted([lbl for lbl, cnt in counts.items() if cnt == max_count], key=lambda x: x.value)
        self.majority_label = candidates[0]
        self.confidence = max_count / len(records)

    def classify(self, text: str) -> SentimentResult:
        if self.majority_label is None:
            raise RuntimeError("classifier must be fitted before inference")

        t0 = time.perf_counter()
        # Fast path inference
        label = self.majority_label
        conf = self.confidence
        latency = (time.perf_counter() - t0) * 1000.0

        return SentimentResult(
            label=label,
            confidence=conf,
            latency_ms=latency,
            tier=self.tier,
            model_name=self.model_name,
        )

    def classify_batch(self, texts: list[str]) -> list[SentimentResult]:
        return [self.classify(t) for t in texts]


class LexiconRuleClassifier:
    _WORD_RE = re.compile(r"\b[a-zA-Z]+\b")

    _POSITIVE_WORDS = frozenset({
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "superb", "love", "best", "positive", "pleased", "nice", "happy",
        "awesome", "perfect", "enjoy", "flawless", "brilliant", "delighted",
    })

    _NEGATIVE_WORDS = frozenset({
        "bad", "terrible", "awful", "horrible", "disgusting", "poor",
        "hate", "worst", "negative", "disappointed", "annoying", "broke",
        "broken", "failed", "slow", "useless", "garbage", "trash", "waste",
    })

    @property
    def tier(self) -> Tier:
        return Tier.TIER_0

    @property
    def model_name(self) -> str:
        return "lexicon_rule_baseline"

    def classify(self, text: str) -> SentimentResult:
        t0 = time.perf_counter()
        words = self._WORD_RE.findall(text.lower()) if text else []

        pos_count = sum(1 for w in words if w in self._POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in self._NEGATIVE_WORDS)

        if pos_count > neg_count:
            label = SentimentLabel.POSITIVE
            diff = pos_count - neg_count
            total = pos_count + neg_count
            confidence = min(1.0, 0.5 + 0.5 * (diff / total))
        elif neg_count > pos_count:
            label = SentimentLabel.NEGATIVE
            diff = neg_count - pos_count
            total = pos_count + neg_count
            confidence = min(1.0, 0.5 + 0.5 * (diff / total))
        else:
            label = SentimentLabel.NEUTRAL
            confidence = 0.50

        latency = (time.perf_counter() - t0) * 1000.0

        return SentimentResult(
            label=label,
            confidence=confidence,
            latency_ms=latency,
            tier=self.tier,
            model_name=self.model_name,
        )

    def classify_batch(self, texts: list[str]) -> list[SentimentResult]:
        return [self.classify(t) for t in texts]
