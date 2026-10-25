import pytest
from sentiments.core import SentimentLabel, Tier
from sentiments.eval.metrics import MetricsCalculator, ClassificationMetrics, EfficiencyMetrics
from sentiments.eval.cost import CostEstimator
from sentiments.eval.reporter import BenchmarkReporter, ModelEvaluation, ComparativeReport


def test_classification_metrics_basic():
    y_true = [
        SentimentLabel.POSITIVE, SentimentLabel.POSITIVE, SentimentLabel.POSITIVE,
        SentimentLabel.NEGATIVE, SentimentLabel.NEGATIVE,
        SentimentLabel.NEUTRAL
    ]
    y_pred = [
        SentimentLabel.POSITIVE, SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE,
        SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL,
        SentimentLabel.POSITIVE
    ]

    metrics = MetricsCalculator.classification_metrics(y_true, y_pred)
    assert isinstance(metrics, ClassificationMetrics)
    assert metrics.accuracy == pytest.approx(0.5)

    # Per class checks:
    # POSITIVE: tp=2, fp=1, fn=1, tn=2, support=3
    pos = metrics.per_class[SentimentLabel.POSITIVE]
    assert metrics.confusion_matrix[SentimentLabel.POSITIVE] == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    assert pos["precision"] == pytest.approx(2 / 3)
    assert pos["recall"] == pytest.approx(2 / 3)
    assert pos["f1"] == pytest.approx(2 / 3)
    assert pos["support"] == 3.0

    # NEGATIVE: tp=1, fp=1, fn=1, tn=3, support=2
    neg = metrics.per_class[SentimentLabel.NEGATIVE]
    assert metrics.confusion_matrix[SentimentLabel.NEGATIVE] == {"tp": 1, "fp": 1, "fn": 1, "tn": 3}
    assert neg["precision"] == pytest.approx(0.5)
    assert neg["recall"] == pytest.approx(0.5)
    assert neg["f1"] == pytest.approx(0.5)
    assert neg["support"] == 2.0

    # NEUTRAL: tp=0, fp=1, fn=1, tn=4, support=1
    neu = metrics.per_class[SentimentLabel.NEUTRAL]
    assert metrics.confusion_matrix[SentimentLabel.NEUTRAL] == {"tp": 0, "fp": 1, "fn": 1, "tn": 4}
    assert neu["precision"] == 0.0
    assert neu["recall"] == 0.0
    assert neu["f1"] == 0.0
    assert neu["support"] == 1.0

    # Aggregates:
    # Macro P, R, F1 = (2/3 + 1/2 + 0) / 3 = 7/18 = 0.3888888888888889
    assert metrics.macro_precision == pytest.approx(7 / 18)
    assert metrics.macro_recall == pytest.approx(7 / 18)
    assert metrics.macro_f1 == pytest.approx(7 / 18)
    # Weighted F1 = (2/3 * 3 + 1/2 * 2 + 0 * 1) / 6 = 3 / 6 = 0.5
    assert metrics.weighted_f1 == pytest.approx(0.5)


def test_classification_metrics_single_support_class():
    # Exactly one sample of positive, predicted correctly -> tp=1, fn=0, fp=0 -> precision=1, recall=1
    metrics = MetricsCalculator.classification_metrics(
        [SentimentLabel.POSITIVE], [SentimentLabel.POSITIVE], labels=[SentimentLabel.POSITIVE]
    )
    assert metrics.per_class[SentimentLabel.POSITIVE]["precision"] == 1.0
    assert metrics.per_class[SentimentLabel.POSITIVE]["recall"] == 1.0
    assert metrics.per_class[SentimentLabel.POSITIVE]["f1"] == 1.0


def test_classification_metrics_empty():
    with pytest.raises(ValueError, match=r"^Input arrays cannot be empty$"):
        MetricsCalculator.classification_metrics([], [])


def test_classification_metrics_length_mismatch():
    with pytest.raises(ValueError, match=r"^Length of y_true and y_pred must match$"):
        MetricsCalculator.classification_metrics([SentimentLabel.POSITIVE], [])


def test_classification_metrics_zero_division():
    # Only positive labels, model predicts only positive
    y_true = [SentimentLabel.POSITIVE, SentimentLabel.POSITIVE]
    y_pred = [SentimentLabel.POSITIVE, SentimentLabel.POSITIVE]

    metrics = MetricsCalculator.classification_metrics(y_true, y_pred, labels=[SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE])
    assert metrics.accuracy == 1.0
    # NEGATIVE class has 0 TP, 0 FP, 0 FN -> precision=0, recall=0, f1=0
    neg_metrics = metrics.per_class[SentimentLabel.NEGATIVE]
    assert neg_metrics["precision"] == 0.0
    assert neg_metrics["recall"] == 0.0
    assert neg_metrics["f1"] == 0.0


def test_efficiency_metrics_basic():
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    eff = MetricsCalculator.efficiency_metrics(latencies, total_duration_s=0.5)
    assert eff.p50_ms == pytest.approx(30.0)
    assert eff.p90_ms == pytest.approx(46.0)
    assert eff.p95_ms == pytest.approx(48.0)
    assert eff.p99_ms == pytest.approx(49.6)
    assert eff.mean_ms == pytest.approx(30.0)
    assert eff.qps == pytest.approx(10.0)  # 5 requests / 0.5s


def test_efficiency_metrics_zero_latency():
    # zero latency list -> total_s = 0 -> qps = 0.0
    eff = MetricsCalculator.efficiency_metrics([0.0])
    assert eff.p50_ms == 0.0
    assert eff.qps == 0.0

    eff_zero_dur = MetricsCalculator.efficiency_metrics([10.0], total_duration_s=0.0)
    assert eff_zero_dur.qps == 0.0


def test_efficiency_metrics_default_qps():
    latencies = [10.0, 10.0]  # sum = 20ms = 0.02s
    eff = MetricsCalculator.efficiency_metrics(latencies)
    assert eff.qps == pytest.approx(2 / 0.02)


def test_efficiency_metrics_empty():
    with pytest.raises(ValueError, match=r"^latencies_ms cannot be empty$"):
        MetricsCalculator.efficiency_metrics([])


def test_efficiency_metrics_negative():
    with pytest.raises(ValueError, match=r"^latencies must be non-negative$"):
        MetricsCalculator.efficiency_metrics([-0.01])


def test_efficiency_metrics_single_item():
    eff = MetricsCalculator.efficiency_metrics([15.0])
    assert eff.p50_ms == 15.0
    assert eff.p99_ms == 15.0
    assert eff.mean_ms == 15.0


def test_percentile_boundary():
    data = [10.0, 20.0, 30.0]
    p100 = MetricsCalculator._percentile(data, 100.0)
    assert p100 == 30.0



def test_cost_estimator_basic():
    # 1_000_000 input at $5/1M, 1_000_000 output at $15/1M -> exactly $20.0
    cost = CostEstimator.estimate(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        input_cost_per_m=5.0,
        output_cost_per_m=15.0,
    )
    assert cost == 20.0

    assert CostEstimator.estimate(0, 0, 0.0, 0.0) == 0.0
    assert CostEstimator.estimate(0, 500, 5.0, 10.0) == pytest.approx(0.005)
    assert CostEstimator.estimate(500, 0, 10.0, 5.0) == pytest.approx(0.005)


def test_cost_estimator_validation():
    with pytest.raises(ValueError, match=r"^tokens and cost rates must be non-negative$"):
        CostEstimator.estimate(-1, 0, 0.0, 0.0)

    with pytest.raises(ValueError, match=r"^tokens and cost rates must be non-negative$"):
        CostEstimator.estimate(0, -1, 0.0, 0.0)

    with pytest.raises(ValueError, match=r"^tokens and cost rates must be non-negative$"):
        CostEstimator.estimate(0, 0, -0.01, 0.0)

    with pytest.raises(ValueError, match=r"^tokens and cost rates must be non-negative$"):
        CostEstimator.estimate(0, 0, 0.0, -0.01)


def test_benchmark_reporter_comparative():
    base_clf = ClassificationMetrics(accuracy=0.5, macro_precision=0.5, macro_recall=0.5, macro_f1=0.5, weighted_f1=0.5, confusion_matrix={}, per_class={})
    base_eff = EfficiencyMetrics(p50_ms=0.2, p90_ms=0.3, p95_ms=0.4, p99_ms=0.5, mean_ms=0.25, qps=5000.0)
    base_eval = ModelEvaluation(model_name="majority-baseline", tier=Tier.TIER_0, classification=base_clf, efficiency=base_eff, estimated_cost_per_1k=0.002)

    cand_clf = ClassificationMetrics(accuracy=0.85, macro_precision=0.85, macro_recall=0.85, macro_f1=0.85, weighted_f1=0.85, confusion_matrix={}, per_class={})
    cand_eff = EfficiencyMetrics(p50_ms=200.0, p90_ms=250.0, p95_ms=280.0, p99_ms=300.0, mean_ms=210.0, qps=5.0)
    cand_eval = ModelEvaluation(model_name="claude-3-5-sonnet", tier=Tier.TIER_1, classification=cand_clf, efficiency=cand_eff, estimated_cost_per_1k=0.010)

    comp = BenchmarkReporter.compare(candidate=cand_eval, baseline=base_eval)
    assert isinstance(comp, ComparativeReport)
    assert comp.candidate == cand_eval
    assert comp.baseline == base_eval
    assert comp.delta_f1 == pytest.approx(0.35)
    assert comp.latency_ratio == pytest.approx(1000.0)  # 200.0 / 0.2
    assert comp.cost_delta == pytest.approx(0.008)  # 0.010 - 0.002 = 0.008


def test_benchmark_reporter_zero_baseline_latency():
    base_clf = ClassificationMetrics(accuracy=0.5, macro_precision=0.5, macro_recall=0.5, macro_f1=0.5, weighted_f1=0.5, confusion_matrix={}, per_class={})
    base_eff = EfficiencyMetrics(p50_ms=0.0, p90_ms=0.0, p95_ms=0.0, p99_ms=0.0, mean_ms=0.0, qps=100.0)
    base_eval = ModelEvaluation(model_name="base", tier=Tier.TIER_0, classification=base_clf, efficiency=base_eff, estimated_cost_per_1k=0.0)

    cand_clf = ClassificationMetrics(accuracy=0.8, macro_precision=0.8, macro_recall=0.8, macro_f1=0.8, weighted_f1=0.8, confusion_matrix={}, per_class={})
    cand_eff = EfficiencyMetrics(p50_ms=10.0, p90_ms=15.0, p95_ms=20.0, p99_ms=25.0, mean_ms=12.0, qps=50.0)
    cand_eval = ModelEvaluation(model_name="cand", tier=Tier.TIER_1, classification=cand_clf, efficiency=cand_eff, estimated_cost_per_1k=0.01)

    comp = BenchmarkReporter.compare(candidate=cand_eval, baseline=base_eval)
    assert comp.latency_ratio == 0.0


def test_benchmark_reporter_markdown_and_json():
    import json
    clf = ClassificationMetrics(accuracy=0.8, macro_precision=0.75, macro_recall=0.70, macro_f1=0.72, weighted_f1=0.78, confusion_matrix={}, per_class={})
    eff = EfficiencyMetrics(p50_ms=10.0, p90_ms=15.0, p95_ms=20.0, p99_ms=25.0, mean_ms=12.0, qps=50.0)
    model_eval = ModelEvaluation(model_name="cand", tier=Tier.TIER_1, classification=clf, efficiency=eff, estimated_cost_per_1k=0.01)

    md = BenchmarkReporter.to_markdown([model_eval])
    expected_md = (
        "| Model | Tier | Accuracy | Macro F1 | p50 Latency (ms) | QPS | Cost / 1k ($) |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        "| cand | Tier.TIER_1 | 0.8000 | 0.7200 | 10.00 | 50.0 | 0.0100 |"
    )
    assert md == expected_md

    base_eval = ModelEvaluation(
        model_name="base", tier=Tier.TIER_0,
        classification=ClassificationMetrics(accuracy=0.5, macro_precision=0.5, macro_recall=0.5, macro_f1=0.5, weighted_f1=0.5, confusion_matrix={}, per_class={}),
        efficiency=EfficiencyMetrics(p50_ms=1.0, p90_ms=1.0, p95_ms=1.0, p99_ms=1.0, mean_ms=1.0, qps=1000.0),
        estimated_cost_per_1k=0.0,
    )
    md_comp = BenchmarkReporter.to_markdown([model_eval], baseline=base_eval)
    expected_md_comp = (
        "| Model | Tier | Accuracy | Macro F1 | Δ F1 | p50 Latency (ms) | Latency Ratio | QPS | Cost / 1k ($) | Δ Cost ($) |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        "| cand | Tier.TIER_1 | 0.8000 | 0.7200 | +0.2200 | 10.00 | 10.0x | 50.0 | 0.0100 | +0.0100 |"
    )
    assert md_comp == expected_md_comp

    json_str = BenchmarkReporter.to_json([model_eval])
    parsed = json.loads(json_str)
    assert len(parsed) == 1
    assert parsed[0] == {
        "model_name": "cand",
        "tier": "Tier.TIER_1",
        "accuracy": 0.8,
        "macro_f1": 0.72,
        "macro_precision": 0.75,
        "macro_recall": 0.70,
        "weighted_f1": 0.78,
        "p50_latency_ms": 10.0,
        "p95_latency_ms": 20.0,
        "qps": 50.0,
        "estimated_cost_per_1k": 0.01,
    }
