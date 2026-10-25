# Sentiments Wiki

`sentiments` is an empirical, multi-tier sentiment analysis and evaluation framework. It benchmarks sentiment classification models across quality (Accuracy, Macro F1), latency ($p_{50}, p_{95}, p_{99}$), throughput ($QPS$), and operational inference cost ($/1k requests) against an empirical **Tier 0 Dumb Baseline**.

---

## Multi-Tier Architecture (Tiers 0–4)

Production NLP teams frequently overpay for hosted frontier LLMs to perform polarity classification tasks that lightweight models or local SLMs can handle with sub-100ms latency and $0.00 marginal cost. `sentiments` categorizes classifiers into five discrete tiers:

| Tier | Identifier | Target Providers / Technologies | Latency ($p_{50}$) | Cost / 1k | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 0** | **Dumb Baseline** | `MajorityClassClassifier`, `LexiconRuleClassifier` | < 0.01 ms | $0.00 | Empirical performance floor; confirms whether ML justifies compute cost. |
| **Tier 1** | **Frontier LLM** | `FrontierLLMClassifier` (Claude 3.5, GPT-4o) | 500–2500 ms | $5.00–$30.00 | Emotion cause analysis, complex sarcasm, taxonomy synthesis. |
| **Tier 2** | **Compact SLM** | `Tier2SLMClassifier` (Apple Metal MLX, Ollama, Llama 3.2 3B, Qwen 2.5 3B) | 40–150 ms | $0.00 (Self-hosted) | Local ABSA, multilingual review triage, private customer data. |
| **Tier 3** | **Fine-Tuned Encoder**| `FineTunedEncoderClassifier` (RoBERTa, DeBERTa-v3) | 10–40 ms | Infra-only | High-throughput sentence polarity classification, support ticket routing. |
| **Tier 4** | **Ultra-Lightweight / ONNX** | `ONNXClassifier` (Quantized ONNX, SetFit, MiniLM) | < 5 ms | Infra-only | Real-time streams, edge CPU inference, clickstream feedback. |

---

## Wiki Navigation Guide

- **[Architecture](file:///Users/mmetawea/Workbench/sentiments/wiki/Architecture.md)**: Hexagonal architecture design, pure domain core vs I/O shell, provider protocols, and system sequence flows.
- **[Performance & Benchmarks](file:///Users/mmetawea/Workbench/sentiments/wiki/Performance.md)**: Empirical Tier 0 Etsy reviews baseline, Tier 2 Apple Silicon (MLX) vs Ollama HTTP loopback benchmarks, memory budgets, and the complete latency-vs-quality trade-off matrix.
- **[Local SLM Developer Guide](file:///Users/mmetawea/Workbench/sentiments/wiki/Local-SLM-Guide.md)**: Developer setup, installation, and zero-shot structured JSON inference with `llama3.2:3b` and `qwen2.5:3b` using MLX Native and Ollama daemon.

---

## Core Principles

1. **Empirical Justification**: Never deploy a complex model without proving it beats a Tier 0 baseline on both Macro F1 and latency-to-dollar efficiency.
2. **Hexagonal Boundaries**: Domain models (`SentimentResult`, `Tier`, `SentimentLabel`) contain zero external I/O or model weights.
3. **Hardware-Native Execution**: Tier 2 runs locally with 4-bit unified memory allocation on Apple Silicon Metal (`mlx-lm`) or local loopback HTTP (`ollama`).
