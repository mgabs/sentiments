Feature: Benchmarking and Comparative Evaluation
  As an ML Engineer
  I want to evaluate sentiment classifiers across performance, latency, and cost
  So that I can objectively benchmark all model tiers against the Tier 0 Dumb Baseline

  Scenario: Calculate classification metrics and confusion matrix
    Given ground truth labels "POSITIVE,NEGATIVE,NEUTRAL,POSITIVE,NEGATIVE"
    And model predictions "POSITIVE,NEGATIVE,POSITIVE,POSITIVE,NEUTRAL"
    When the metrics calculator computes classification metrics
    Then accuracy should be 0.6
    And macro F1 score should be greater than 0.4
    And the confusion matrix should have correct true and false counts

  Scenario: Calculate latency percentiles and throughput
    Given a list of request latencies "[10.0, 15.0, 20.0, 25.0, 100.0]" in milliseconds
    When the efficiency metrics are computed
    Then p50 latency should be 20.0 ms
    And p95 latency should be 85.0 ms
    And throughput QPS should be positive

  Scenario: Estimate operational inference costs
    Given 1000 input tokens and 200 output tokens
    And cost rates of 5.0 dollars per million input tokens and 15.0 dollars per million output tokens
    When the cost estimator calculates total cost
    Then the total cost should be 0.008 dollars

  Scenario: Comparative delta reporting against Tier 0 baseline
    Given a Tier 0 baseline report with F1 0.50, latency 0.2 ms, and cost 0.0 dollars
    And a candidate Tier 1 report with F1 0.85, latency 200.0 ms, and cost 0.008 dollars
    When the benchmark reporter generates a comparative report
    Then delta F1 should be 0.35
    And latency ratio should be 1000.0
    And cost delta should be 0.008 dollars

  Scenario: Metric calculation handles length mismatch and empty inputs
    Given ground truth labels "POSITIVE,NEGATIVE"
    And model predictions "POSITIVE"
    When metrics computation is attempted with mismatched lengths
    Then a value error should be raised
