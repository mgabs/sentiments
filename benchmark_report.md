# Multi-Tier Sentiment Benchmark Report

Dataset: `nlp-datasets/trustpilot-reviews` (137 test samples)

| Model | Tier | Accuracy | Macro F1 | Δ F1 | p50 Latency (ms) | Latency Ratio | QPS | Cost / 1k ($) | Δ Cost ($) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MajorityClassClassifier | Tier.TIER_0 | 0.9270 | 0.3207 | +0.0000 | 0.00 | 1.0x | 957205.3 | 0.0000 | +0.0000 |
| LexiconRuleClassifier | Tier.TIER_0 | 0.4745 | 0.2828 | -0.0379 | 0.01 | 270.9x | 68737.7 | 0.0000 | +0.0000 |
| Ollama (llama3.2:3b) | Tier.TIER_2 | 0.9781 | 0.6257 | +0.3050 | 1245.20 | 29649894.5x | 0.8 | 0.0000 | +0.0000 |
| Ollama (qwen2.5:3b) | Tier.TIER_2 | 0.9051 | 0.5942 | +0.2735 | 1139.98 | 27144371.9x | 0.8 | 0.0000 | +0.0000 |
| MLX (Llama-3.2-3B-Instruct-4bit) | Tier.TIER_2 | 0.9781 | 0.6257 | +0.3050 | 2278.88 | 54263122.8x | 0.4 | 0.0000 | +0.0000 |
| MLX (Qwen2.5-3B-Instruct-4bit) | Tier.TIER_2 | 0.9197 | 0.5818 | +0.2610 | 2159.10 | 51410974.8x | 0.5 | 0.0000 | +0.0000 |
| ONNX (setfit-minilm-int8) | Tier.TIER_4 | 0.9489 | 0.5021 | +0.1814 | 1.87 | 44440.9x | 479.2 | 0.0000 | +0.0000 |

### Column Definitions
- **Model**: Evaluated classifier name.
- **Tier**: Architectural tier (Tier 0: Rule-based heuristics, Tier 2: Local SLM, Tier 4: Quantized ONNX CPU).
- **Accuracy**: Proportion of correct predictions over total samples.
- **Macro F1**: Unweighted mean of F1-scores across all classes.
- **Δ F1**: Difference in Macro F1 relative to baseline (`Macro F1(Candidate) - Macro F1(Baseline)`).
- **p50 Latency (ms)**: Median inference latency per sample in milliseconds.
- **Latency Ratio**: Relative latency multiplier compared to baseline (`p50(Candidate) / p50(Baseline)`).
- **QPS**: Throughput measured in queries processed per second.
- **Cost / 1k ($)**: Estimated operational cost in USD per 1,000 predictions.
- **Δ Cost ($)**: Cost delta per 1,000 predictions relative to baseline (`Cost(Candidate) - Cost(Baseline)`).
