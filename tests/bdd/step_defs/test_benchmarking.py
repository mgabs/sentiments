import json
import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from sentiments.core import SentimentLabel, Tier
from sentiments.eval.metrics import MetricsCalculator, ClassificationMetrics, EfficiencyMetrics
from sentiments.eval.cost import CostEstimator
from sentiments.eval.reporter import BenchmarkReporter, ModelEvaluation, ComparativeReport

scenarios("../features/benchmarking.feature")


@pytest.fixture
def eval_context():
    return {}


@given(parsers.parse('ground truth labels "{labels}"'), target_fixture="eval_context")
def set_ground_truth(eval_context, labels):
    eval_context["y_true"] = [SentimentLabel(l.strip()) for l in labels.split(",")]
    return eval_context


@given(parsers.parse('model predictions "{labels}"'))
def set_predictions(eval_context, labels):
    eval_context["y_pred"] = [SentimentLabel(l.strip()) for l in labels.split(",")]


@when("the metrics calculator computes classification metrics")
def compute_metrics(eval_context):
    eval_context["metrics"] = MetricsCalculator.classification_metrics(
        eval_context["y_true"], eval_context["y_pred"]
    )


@then(parsers.parse("accuracy should be {expected:g}"))
def assert_accuracy(eval_context, expected):
    assert eval_context["metrics"].accuracy == pytest.approx(expected)


@then(parsers.parse("macro F1 score should be greater than {threshold:g}"))
def assert_macro_f1(eval_context, threshold):
    assert eval_context["metrics"].macro_f1 > threshold


@then("the confusion matrix should have correct true and false counts")
def assert_confusion_matrix(eval_context):
    cm = eval_context["metrics"].confusion_matrix
    assert isinstance(cm, dict)
    assert SentimentLabel.POSITIVE in cm


@given(parsers.parse('a list of request latencies "{latencies_json}" in milliseconds'), target_fixture="eval_context")
def set_latencies(eval_context, latencies_json):
    eval_context["latencies"] = json.loads(latencies_json)
    return eval_context


@when("the efficiency metrics are computed")
def compute_efficiency(eval_context):
    eval_context["eff"] = MetricsCalculator.efficiency_metrics(eval_context["latencies"])


@then(parsers.parse("p50 latency should be {expected:g} ms"))
def assert_p50(eval_context, expected):
    assert eval_context["eff"].p50_ms == pytest.approx(expected)


@then(parsers.parse("p95 latency should be {expected:g} ms"))
def assert_p95(eval_context, expected):
    assert eval_context["eff"].p95_ms == pytest.approx(expected)


@then("throughput QPS should be positive")
def assert_throughput(eval_context):
    assert eval_context["eff"].qps > 0.0


@given(parsers.parse("{input_tokens:d} input tokens and {output_tokens:d} output tokens"), target_fixture="eval_context")
def set_tokens(eval_context, input_tokens, output_tokens):
    eval_context["input_tokens"] = input_tokens
    eval_context["output_tokens"] = output_tokens
    return eval_context


@given(parsers.parse("cost rates of {in_rate:g} dollars per million input tokens and {out_rate:g} dollars per million output tokens"))
def set_cost_rates(eval_context, in_rate, out_rate):
    eval_context["in_rate"] = in_rate
    eval_context["out_rate"] = out_rate


@when("the cost estimator calculates total cost")
def compute_cost(eval_context):
    eval_context["cost"] = CostEstimator.estimate(
        eval_context["input_tokens"],
        eval_context["output_tokens"],
        eval_context["in_rate"],
        eval_context["out_rate"],
    )


@then(parsers.parse("the total cost should be {expected:g} dollars"))
def assert_total_cost(eval_context, expected):
    assert eval_context["cost"] == pytest.approx(expected)


@given(parsers.parse("a Tier 0 baseline report with F1 {f1:g}, latency {lat:g} ms, and cost {cost:g} dollars"), target_fixture="eval_context")
def set_baseline_report(eval_context, f1, lat, cost):
    base_clf = ClassificationMetrics(accuracy=f1, macro_precision=f1, macro_recall=f1, macro_f1=f1, weighted_f1=f1, confusion_matrix={}, per_class={})
    base_eff = EfficiencyMetrics(p50_ms=lat, p90_ms=lat, p95_ms=lat, p99_ms=lat, mean_ms=lat, qps=1000.0)
    eval_context["baseline"] = ModelEvaluation(model_name="majority-baseline", tier=Tier.TIER_0, classification=base_clf, efficiency=base_eff, estimated_cost_per_1k=cost)
    return eval_context


@given(parsers.parse("a candidate Tier 1 report with F1 {f1:g}, latency {lat:g} ms, and cost {cost:g} dollars"))
def set_candidate_report(eval_context, f1, lat, cost):
    cand_clf = ClassificationMetrics(accuracy=f1, macro_precision=f1, macro_recall=f1, macro_f1=f1, weighted_f1=f1, confusion_matrix={}, per_class={})
    cand_eff = EfficiencyMetrics(p50_ms=lat, p90_ms=lat, p95_ms=lat, p99_ms=lat, mean_ms=lat, qps=5.0)
    eval_context["candidate"] = ModelEvaluation(model_name="claude-3-5-sonnet", tier=Tier.TIER_1, classification=cand_clf, efficiency=cand_eff, estimated_cost_per_1k=cost)


@when("the benchmark reporter generates a comparative report")
def generate_comp_report(eval_context):
    eval_context["comparison"] = BenchmarkReporter.compare(
        candidate=eval_context["candidate"],
        baseline=eval_context["baseline"],
    )


@then(parsers.parse("delta F1 should be {expected:g}"))
def assert_delta_f1(eval_context, expected):
    assert eval_context["comparison"].delta_f1 == pytest.approx(expected)


@then(parsers.parse("latency ratio should be {expected:g}"))
def assert_latency_ratio(eval_context, expected):
    assert eval_context["comparison"].latency_ratio == pytest.approx(expected)


@then(parsers.parse("cost delta should be {expected:g} dollars"))
def assert_cost_delta(eval_context, expected):
    assert eval_context["comparison"].cost_delta == pytest.approx(expected)


@when("metrics computation is attempted with mismatched lengths")
def attempt_mismatch(eval_context):
    with pytest.raises(ValueError) as exc:
        MetricsCalculator.classification_metrics(eval_context["y_true"], eval_context["y_pred"])
    eval_context["error"] = exc.value


@then("a value error should be raised")
def assert_value_error(eval_context):
    assert isinstance(eval_context["error"], ValueError)
