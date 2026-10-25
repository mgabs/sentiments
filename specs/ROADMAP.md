# Sentiments: Product & Technical Roadmap

## 1. High-Level Vision & North Star

Build an empirical, spec-driven sentiment evaluation and inference platform comparing model architectures across quality, latency, throughput, and operational economics.

### Guiding Principles
- **Dumb Baselines First**: Never deploy or justify an expensive, slow model without proving a statistically significant delta ($\Delta F1 > 0$) over the free, sub-millisecond Tier 0 floor.
- **Spec- & Test-Driven (SDTRD)**: Every capability begins with Gherkin behavior specs (100% behavior coverage), followed by unit tests ($\ge 95\%$ line/branch coverage), and mutation verification ($100\%$ kill score).
- **Pareto-Optimal Selection**: Select models based on real-world engineering trade-offs (Cost per 1k calls, $p_{95}$ latency SLA, QPS) rather than raw benchmark vanity scores.

---

## 2. Multi-Tier Architecture Matrix

```mermaid
quadrantChart
    title Sentiment Classification: Cost vs. Latency Trade-Offs
    x-axis Low Latency --> High Latency
    y-axis Low Cost --> High Cost
    quadrant-1 Unviable / Inefficient
    quadrant-2 Frontier Reasoning (Tier 1)
    quadrant-3 Floor Baseline (Tier 0), Edge ONNX (Tier 4)
    quadrant-4 Self-Hosted SLM (Tier 2), Fine-Tuned Encoders (Tier 3)
    "Majority Baseline": [0.05, 0.05]
    "Lexicon Rule": [0.10, 0.05]
    "Quantized ONNX": [0.15, 0.10]
    "Fine-Tuned RoBERTa": [0.35, 0.20]
    "Compact SLM (vLLM)": [0.55, 0.35]
    "Frontier LLM (GPT-4o/Claude)": [0.90, 0.85]
```

| Tier | Paradigm | Target Latency ($p_{50}$) | Target Cost / 1k | Best Suited For | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 0** | Dumb Baseline (Majority Class, Lexicon Polarity) | $< 0.01\text{ ms}$ | $\$0.00$ | Floor baseline; offline sanity check | **Completed** |
| **Tier 1** | Frontier Zero/Few-Shot Reasoning (Claude 3.5, GPT-4o) | $500–2500\text{ ms}$ | $\$5.00–\$30.00$ | Implicit sarcasm, emotion causes, taxonomy discovery | **Backlog** |
| **Tier 2** | Distilled Compact SLMs (Llama 3 8B, Qwen 2.5, Phi-4) | $50–150\text{ ms}$ | $\$0.05–\$0.20$ | Self-hosted batch inference, ABSA, multi-lingual | **Backlog** |
| **Tier 3** | Specialized Fine-Tuned Encoders (RoBERTa, DeBERTa) | $10–40\text{ ms}$ | Fixed Infra | High-volume ticket classification, standard polarity | **Backlog** |
| **Tier 4** | Ultra-Lightweight Edge / ONNX (Quantized, SetFit) | $< 5\text{ ms}$ | Negligible CPU | Real-time streaming, edge devices, high-QPS firehoses | **Backlog** |

---

## 3. Milestone Feature Pipeline (Input for `specs/features/`)

### Milestone 1: Core Foundation & Dumb Baseline (`feat/01`) — [COMPLETED]
- [x] Canonical domain contracts (`SentimentLabel`, `Tier`, `SentimentResult`, `SentimentClassifier`).
- [x] Dataset ingestion pipeline discovering heterogeneous CSV files from `nlp-datasets/`.
- [x] Text sanitization, metadata extraction (`product`, `category`, `platform`, `language`).
- [x] Stratified deterministic train/val/test splitter (70/15/15).
- [x] Tier 0 classifiers: `MajorityClassClassifier` & `LexiconRuleClassifier`.
- [x] Evaluation engine: Accuracy, Macro F1, Precision, Recall, Confusion Matrix, Latency percentiles, QPS, Cost estimator, Markdown & JSON reports.
- [x] 100% BDD behavior coverage, 99% branch coverage, 100% mutation kill score on core logic.

---

### Milestone 2: Live Tier 1 Frontier LLM Provider Integration (`feat/02`)
*Target Feature*: `specs/features/02-tier1-frontier-provider`
- **Objective**: Connect `FrontierLLMClassifier` to real API endpoints (Anthropic Claude 3.5 Sonnet, OpenAI GPT-4o, Google Gemini 1.5).
- **Key Capabilities**:
  - Environment-based API credentials (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).
  - Structured output schema enforcement via prompt engineering or native JSON mode.
  - Token counting, rate-limiting, and per-call cost tracking.
  - Batching and retry logic with exponential backoff on transient errors.
- **Evaluation Target**: Measure accuracy uplift ($\Delta F1$) on customer reviews versus Tier 0 to identify if prompt cost is justified.

---

### Milestone 3: Live Tier 4 Quantized ONNX & SetFit CPU Runtime (`feat/03`)
*Target Feature*: `specs/features/03-tier4-onnx-runtime`
- **Objective**: Productionize `ONNXClassifier` with pre-converted INT8 quantized transformer models running locally on CPU.
- **Key Capabilities**:
  - Packaging an optimized quantized DistilBERT / MiniLM model artifact (`.onnx`).
  - Native CPU execution via `onnxruntime` with sub-5ms latency budgets.
  - SetFit few-shot sentence transformer pipeline for rapid fine-tuning on small datasets.
  - End-to-end memory footprint profiling.
- **Evaluation Target**: Deliver $> 0.85$ Macro F1 with $< 5\text{ms}$ latency and 0 external network calls.

---

### Milestone 4: Live Tier 3 Fine-Tuned Transformer Pipeline (`feat/04`)
*Target Feature*: `specs/features/04-tier3-encoder-pipeline`
- **Objective**: Full training and inference loop for specialized encoder models (RoBERTa-base, DeBERTa-v3).
- **Key Capabilities**:
  - Fine-tuning script consuming `StratifiedSplitter` training sets from `nlp-datasets/`.
  - Checkpoint serialization and evaluation on validation sets with early stopping.
  - Integration with `FineTunedEncoderClassifier` for batch inference.
  - Hardware acceleration support (Apple Silicon MPS / CUDA / CPU fallback).
- **Evaluation Target**: Establish domain-specific accuracy ceiling on customer reviews.

---

### Milestone 5: Live Tier 2 Self-Hosted SLM & Distillation (`feat/05`)
*Target Feature*: `specs/features/05-tier2-slm-distillation`
- **Objective**: Interface with local/hosted OpenAI-compatible inference servers (vLLM, Ollama) and implement teacher-student distillation.
- **Key Capabilities**:
  - `CompactSLMClassifier` client for Ollama / vLLM endpoints.
  - Synthetic dataset generation: Use Tier 1 models to label unannotated review corpora.
  - LoRA fine-tuning recipes for Llama 3 8B / Qwen 2.5 7B.
- **Evaluation Target**: Achieve $\ge 90\%$ of Tier 1 reasoning quality at a fraction of the inference latency and zero API per-call fees.

---

### Milestone 6: Pareto Frontier Engine & Automated Tier Selection (`feat/06`)
*Target Feature*: `specs/features/06-pareto-selection-engine`
- **Objective**: Automated model recommendation engine based on user-specified operational constraints.
- **Key Capabilities**:
  - Constraint solver: e.g. "Find best F1 model with $p_{95} < 20\text{ms}$ and $\text{Cost} < \$0.01/\text{1k}$".
  - Multi-tier Pareto frontier visualization (ASCII / Markdown / HTML scatter plots).
  - Production fallback cascading router: Tier 0 $\rightarrow$ Tier 4 $\rightarrow$ Tier 1 on low-confidence triggers.

---

## 4. Dataset Coverage Plan

| Dataset Corpus | Domain | Review Volume | Target Usage |
| :--- | :--- | :--- | :--- |
| `trustpilot-reviews/` | Multi-domain consumer reviews | ~1,000 | General consumer sentiment baseline |
| `etsy-reviews/` | E-commerce / handmade goods | ~100 | Few-shot product review evaluation |
| `udemy-reviews/` | Online course education feedback | ~1,000 | Aspect-based and instructional reviews |
| `app-store-reviews/` | Mobile iOS apps | ~5,000 | Bug reporting vs praise classification |
| `training-sets/` | Curated multi-source sentiment | ~10,000 | Primary training & distillation corpus |
