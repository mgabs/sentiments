# Specification: Benchmarking & Comparison Framework

## 1. Overview
Defines requirements for automated evaluation, metric computation, tier comparison, and reporting relative to Tier 0 baseline performance.

## 2. Requirements & Capabilities

### Capability: Metrics Computation
- **REQ-BENCH-1**: The evaluator MUST compute standard classification metrics against ground truth:
  - Accuracy
  - Macro and Weighted F1-score
  - Macro Precision and Recall
  - Multi-class Confusion Matrix (True Positive, False Positive, True Negative, False Negative per class)
- **REQ-BENCH-2**: The evaluator MUST measure runtime efficiency:
  - Latency percentiles: p50, p95, p99 (in ms).
  - Throughput: Queries per second (QPS).
  - Peak memory / resource footprint where measurable.
- **REQ-BENCH-3**: The evaluator MUST estimate operational cost:
  - Input & output token cost per 1K calls for API tiers.
  - Compute infrastructure cost estimation for local/self-hosted tiers.

### Capability: Multi-Tier Comparative Reporting
- **REQ-BENCH-4**: The benchmarking framework MUST output a comparison table ranking models against the Tier 0 Dumb Baseline:
  - Metric delta: `Δ F1 = F1(Tier_N) - F1(Tier_0)`
  - Latency multiplier: `Latency_Ratio = Latency(Tier_N) / Latency(Tier_0)`
  - Cost ratio: `Cost_Delta = Cost(Tier_N) - Cost(Tier_0)`
- **REQ-BENCH-5**: The runner MUST serialize evaluation outputs into JSON and formatted Markdown tables.

## 3. Assumptions
- **ASM-BENCH-1**: Evaluations are run against a held-out test split created deterministically by the dataset pipeline.
- **ASM-BENCH-2**: Cost calculations for Tier 1 rely on published rates (e.g., $5.00/1M input tokens, $15.00/1M output tokens) and average prompt token count.
- **ASM-BENCH-3**: Tier 0 operational cost is assumed $0.00 for marginal requests.
- **ASM-BENCH-4**: Baseline comparisons measure delta improvement over Tier 0 `LexiconRuleClassifier` (or `MajorityClassClassifier`).

## 4. Edge Cases & Error Handling
- **Zero Division**: Guarding against zero division when computing precision/recall for unpredicted classes.
- **Mismatched Dataset Sizes**: Validating that predictions match ground-truth array lengths before metric calculation.
- **Timeout Handling**: Per-record timeout during benchmark runs to avoid hanging on unresponsive external tiers.
