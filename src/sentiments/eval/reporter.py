import json
from dataclasses import dataclass, asdict
from typing import Sequence
from sentiments.core import Tier
from sentiments.eval.metrics import ClassificationMetrics, EfficiencyMetrics


@dataclass(frozen=True)
class ModelEvaluation:
    model_name: str
    tier: Tier
    classification: ClassificationMetrics
    efficiency: EfficiencyMetrics
    estimated_cost_per_1k: float = 0.0


@dataclass(frozen=True)
class ComparativeReport:
    candidate: ModelEvaluation
    baseline: ModelEvaluation
    delta_f1: float
    latency_ratio: float
    cost_delta: float


class BenchmarkReporter:
    @staticmethod
    def compare(candidate: ModelEvaluation, baseline: ModelEvaluation) -> ComparativeReport:
        delta_f1 = candidate.classification.macro_f1 - baseline.classification.macro_f1
        latency_ratio = (
            candidate.efficiency.p50_ms / baseline.efficiency.p50_ms
            if baseline.efficiency.p50_ms > 0.0
            else 0.0
        )
        cost_delta = candidate.estimated_cost_per_1k - baseline.estimated_cost_per_1k

        return ComparativeReport(
            candidate=candidate,
            baseline=baseline,
            delta_f1=delta_f1,
            latency_ratio=latency_ratio,
            cost_delta=cost_delta,
        )

    @classmethod
    def to_markdown(
        cls,
        evaluations: Sequence[ModelEvaluation],
        baseline: ModelEvaluation | None = None,
    ) -> str:
        lines: list[str] = []
        if baseline is not None:
            lines.append(
                "| Model | Tier | Accuracy | Macro F1 | Δ F1 | p50 Latency (ms) | Latency Ratio | QPS | Cost / 1k ($) | Δ Cost ($) |"
            )
            lines.append(
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
            )
            for ev in evaluations:
                comp = cls.compare(candidate=ev, baseline=baseline)
                lines.append(
                    f"| {ev.model_name} | {ev.tier} | {ev.classification.accuracy:.4f} | "
                    f"{ev.classification.macro_f1:.4f} | {comp.delta_f1:+.4f} | "
                    f"{ev.efficiency.p50_ms:.2f} | {comp.latency_ratio:.1f}x | "
                    f"{ev.efficiency.qps:.1f} | {ev.estimated_cost_per_1k:.4f} | "
                    f"{comp.cost_delta:+.4f} |"
                )
        else:
            lines.append(
                "| Model | Tier | Accuracy | Macro F1 | p50 Latency (ms) | QPS | Cost / 1k ($) |"
            )
            lines.append(
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
            )
            for ev in evaluations:
                lines.append(
                    f"| {ev.model_name} | {ev.tier} | {ev.classification.accuracy:.4f} | "
                    f"{ev.classification.macro_f1:.4f} | {ev.efficiency.p50_ms:.2f} | "
                    f"{ev.efficiency.qps:.1f} | {ev.estimated_cost_per_1k:.4f} |"
                )
        return "\n".join(lines)

    @staticmethod
    def to_json(evaluations: Sequence[ModelEvaluation]) -> str:
        data = []
        for ev in evaluations:
            data.append({
                "model_name": ev.model_name,
                "tier": str(ev.tier),
                "accuracy": ev.classification.accuracy,
                "macro_f1": ev.classification.macro_f1,
                "macro_precision": ev.classification.macro_precision,
                "macro_recall": ev.classification.macro_recall,
                "weighted_f1": ev.classification.weighted_f1,
                "p50_latency_ms": ev.efficiency.p50_ms,
                "p95_latency_ms": ev.efficiency.p95_ms,
                "qps": ev.efficiency.qps,
                "estimated_cost_per_1k": ev.estimated_cost_per_1k,
            })
        return json.dumps(data)
