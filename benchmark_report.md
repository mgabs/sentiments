# Tier 0 Baseline Benchmark Report

Dataset: `nlp-datasets/etsy-reviews` (137 test samples)

| Model | Tier | Accuracy | Macro F1 | Δ F1 | p50 Latency (ms) | Latency Ratio | QPS | Cost / 1k ($) | Δ Cost ($) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MajorityClassClassifier | Tier.TIER_0 | 0.9270 | 0.3207 | +0.0000 | 0.00 | 1.0x | 1035595.8 | 0.0000 | +0.0000 |
| LexiconRuleClassifier | Tier.TIER_0 | 0.4745 | 0.2828 | -0.0379 | 0.01 | 261.9x | 69918.8 | 0.0000 | +0.0000 |
