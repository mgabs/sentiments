from dataclasses import dataclass, field
from typing import Sequence
from sentiments.core import SentimentLabel


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    confusion_matrix: dict[SentimentLabel, dict[str, int]]
    per_class: dict[SentimentLabel, dict[str, float]]


@dataclass(frozen=True)
class EfficiencyMetrics:
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    qps: float


class MetricsCalculator:
    @staticmethod
    def classification_metrics(
        y_true: Sequence[SentimentLabel],
        y_pred: Sequence[SentimentLabel],
        labels: Sequence[SentimentLabel] | None = None,
    ) -> ClassificationMetrics:
        if len(y_true) != len(y_pred):
            raise ValueError("Length of y_true and y_pred must match")
        if not y_true:
            raise ValueError("Input arrays cannot be empty")

        if labels is None:
            labels = list(SentimentLabel)

        n = len(y_true)
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n

        confusion_matrix: dict[SentimentLabel, dict[str, int]] = {}
        per_class: dict[SentimentLabel, dict[str, float]] = {}

        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        supports: list[int] = []

        for lbl in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
            tn = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p != lbl)
            support = sum(1 for t in y_true if t == lbl)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            confusion_matrix[lbl] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
            per_class[lbl] = {"precision": precision, "recall": recall, "f1": f1, "support": float(support)}

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(support)

        macro_precision = sum(precisions) / len(labels)
        macro_recall = sum(recalls) / len(labels)
        macro_f1 = sum(f1s) / len(labels)

        total_support = sum(supports)
        weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / total_support

        return ClassificationMetrics(
            accuracy=accuracy,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            confusion_matrix=confusion_matrix,
            per_class=per_class,
        )

    @staticmethod
    def _percentile(sorted_data: list[float], p: float) -> float:
        n = len(sorted_data)
        rank = (p / 100.0) * (n - 1)
        k = int(rank)
        d = rank - k
        if k + 1 < n:
            return sorted_data[k] + d * (sorted_data[k + 1] - sorted_data[k])
        return float(sorted_data[-1])

    @classmethod
    def efficiency_metrics(
        cls,
        latencies_ms: Sequence[float],
        total_duration_s: float | None = None,
    ) -> EfficiencyMetrics:
        if not latencies_ms:
            raise ValueError("latencies_ms cannot be empty")
        if any(lat < 0 for lat in latencies_ms):
            raise ValueError("latencies must be non-negative")

        sorted_lats = sorted(latencies_ms)
        p50 = cls._percentile(sorted_lats, 50.0)
        p90 = cls._percentile(sorted_lats, 90.0)
        p95 = cls._percentile(sorted_lats, 95.0)
        p99 = cls._percentile(sorted_lats, 99.0)
        mean_ms = sum(sorted_lats) / len(sorted_lats)

        if total_duration_s is not None:
            qps = len(latencies_ms) / total_duration_s if total_duration_s > 0 else 0.0
        else:
            total_s = sum(latencies_ms) / 1000.0
            qps = len(latencies_ms) / total_s if total_s > 0 else 0.0

        return EfficiencyMetrics(
            p50_ms=p50,
            p90_ms=p90,
            p95_ms=p95,
            p99_ms=p99,
            mean_ms=mean_ms,
            qps=qps,
        )
