# Sentiments: Spec-Driven Multi-Tier Sentiment Benchmarking

Pluggable sentiment analysis benchmarking framework comparing model architectures across quality (Accuracy, Macro F1), latency ($p_{50}, p_{95}, p_{99}$), throughput ($QPS$), and operational cost ($/1k requests) against an empirical **Tier 0 Dumb Baseline**.

Built following strict **Spec- and Test-Driven Development (STDD)** via OpenSpec: 100% BDD behavior coverage, 99% branch coverage, and 100% mutation kill score on core logic.

---

## Model Tiers

| Tier | Description | Latency ($p_{50}$) | Cost / 1k | Best Suited For |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 0: Dumb Baseline** | `MajorityClassClassifier`, `LexiconRuleClassifier` | < 0.01 ms | $0.00 | Performance floor; proves whether complex models justify compute cost. |
| **Tier 1: Frontier LLM** | `FrontierLLMClassifier` (Claude 3.5, GPT-4o) | 500–2500 ms | $5–$30 | Emotion cause analysis, complex sarcasm, unstandardized taxonomy synthesis. |
| **Tier 2: Compact SLM** | `CompactSLMClassifier` (vLLM / Ollama, Llama 8B, Qwen) | 50–150 ms | $0.05–$0.20 | Structured ABSA, multilingual feedback, high-accuracy triage. |
| **Tier 3: Fine-Tuned Encoder** | `FineTunedEncoderClassifier` (RoBERTa, DeBERTa) | 10–40 ms | Infra-only | High-throughput sentence polarity classification, support ticket routing. |
| **Tier 4: Ultra-Lightweight / ONNX** | `ONNXClassifier` (Quantized ONNX, SetFit) | < 5 ms | Infra-only | Real-time streams, CPU edge inference, clickstream feedback. |

---

## Project Structure

```
├── benchmark_report.md        # Live Tier 0 benchmark report on customer reviews
├── ROADMAP.md                 # Product & technical roadmap for upcoming features
├── nlp-datasets/              # Raw labeled feedback corpora (Etsy, Trustpilot, etc.)
├── scripts/
│   └── benchmark_baseline.py  # End-to-end evaluation runner on nlp-datasets/
├── specs/
│   ├── WORKFLOW.md            # SDTRD/STDD lifecycle, review gates, anti-fraud test rules
│   ├── archive/               # Archived completed features (proposals, tasks, specs)
│   ├── features/              # Active feature proposals and task breakdowns
│   └── living/                # Canonical living specifications (dataset, engine, eval)
├── src/sentiments/
│   ├── core/                  # Core domain models, protocols (SentimentResult, Tier)
│   ├── dataset/               # Ingestion, cleaning, metadata extraction, stratified splitting
│   ├── eval/                  # Metrics calculation, cost estimation, Markdown/JSON reporting
│   └── providers/             # Classifier implementations across Tiers 0–4
└── tests/
    ├── bdd/                   # Gherkin behavior specifications and step definitions
    └── unit/                  # Comprehensive unit tests with branch & mutation coverage
```

---

## Quickstart

### Prerequisites
- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv)

### Installation
```bash
make install    # or: uv sync
```

### Run Tests & Quality Gates
```bash
# 1. Run all unit and BDD tests
make test

# 2. Run test suite with branch coverage (enforcing >= 95%)
make coverage

# 3. Run mutation testing (100% kill score on core logic)
make mutation

# 4. Run baseline benchmark
make benchmark
```

### Run Baseline Benchmark
Evaluates `MajorityClassClassifier` and `LexiconRuleClassifier` on real customer reviews:
```bash
uv run python scripts/benchmark_baseline.py
```

Sample output:
```markdown
| Model | Tier | Accuracy | Macro F1 | Δ F1 | p50 Latency (ms) | Latency Ratio | QPS | Cost / 1k ($) | Δ Cost ($) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MajorityClassClassifier | Tier.TIER_0 | 0.9270 | 0.3207 | +0.0000 | 0.00 | 1.0x | 1061332.7 | 0.0000 | +0.0000 |
| LexiconRuleClassifier | Tier.TIER_0 | 0.4745 | 0.2828 | -0.0379 | 0.01 | 255.0x | 70239.9 | 0.0000 | +0.0000 |
```

---

## Usage Examples

### 1. Ingest & Split Datasets
```python
from sentiments.dataset import DatasetLoader, StratifiedSplitter

# Auto-discovers CSV files, detects delimiters, extracts metadata
records = DatasetLoader("nlp-datasets/trustpilot-reviews").load_all()

# Deterministic stratified 70/15/15 train/val/test split
splitter = StratifiedSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
split = splitter.split(records)
```

### 2. Classify with Tier 0 Baseline
```python
from sentiments.providers.tier0 import MajorityClassClassifier, LexiconRuleClassifier

# Majority class classifier
maj_clf = MajorityClassClassifier()
maj_clf.fit(split.train)
res = maj_clf.classify("Superb customer service!")
print(res.label, res.confidence, res.latency_ms)

# Lexicon rule-based classifier (offline keyword polarity)
lex_clf = LexiconRuleClassifier()
res = lex_clf.classify("Worst experience, broken item.")
print(res.label, res.confidence, res.latency_ms)
```

### 3. Evaluate & Compare Against Baseline
```python
from sentiments.eval import MetricsCalculator, BenchmarkReporter, ModelEvaluation
from sentiments.core import Tier

# Compute classification & efficiency metrics
clf_metrics = MetricsCalculator.classification_metrics(y_true, y_pred)
eff_metrics = MetricsCalculator.efficiency_metrics(latencies_ms)

# Generate comparative Markdown table against baseline
report = BenchmarkReporter.to_markdown([candidate_eval], baseline=baseline_eval)
print(report)
```

---

## Specifications & STDD Workflow

All architectural decisions and behavior specifications are tracked in [specs/](file:///Users/mmetawea/Workbench/sentiments/specs):
- [ROADMAP.md](file:///Users/mmetawea/Workbench/sentiments/specs/ROADMAP.md): High-level product & engineering roadmap with milestone features.
- [specs/WORKFLOW.md](file:///Users/mmetawea/Workbench/sentiments/specs/WORKFLOW.md): SDTRD/STDD protocol, human review gates, and anti-fraud testing guidelines.
- [specs/living/dataset-pipeline/spec.md](file:///Users/mmetawea/Workbench/sentiments/specs/living/dataset-pipeline/spec.md): Dataset ingestion and normalization requirements.
- [specs/living/sentiment-engine/spec.md](file:///Users/mmetawea/Workbench/sentiments/specs/living/sentiment-engine/spec.md): Pluggable multi-tier inference engine contracts.
- [specs/living/benchmarking/spec.md](file:///Users/mmetawea/Workbench/sentiments/specs/living/benchmarking/spec.md): Metric calculation, latency percentiles, and cost estimation specs.
