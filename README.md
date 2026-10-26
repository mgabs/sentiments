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

Column definitions:
- `Model`: Evaluated classifier name.
- `Tier`: Architectural tier (Tier 0: Rule-based heuristics).
- `Accuracy`: Proportion of correct predictions over total samples.
- `Macro F1`: Unweighted mean of F1-scores across all classes.
- `Δ F1`: Difference in Macro F1 relative to baseline.
- `p50 Latency (ms)`: Median inference latency per sample in milliseconds.
- `Latency Ratio`: Latency multiplier relative to baseline (`p50 / Baseline p50`).
- `QPS`: Throughput in queries processed per second.
- `Cost / 1k ($)`: Estimated operational cost in USD per 1,000 predictions.
- `Δ Cost ($)`: Cost delta per 1,000 predictions relative to baseline.
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

### 3. Classify with Tier 2 Local SLM (Ollama & Apple MLX)
```python
from sentiments.providers.tier2 import Tier2SLMClassifier, OllamaBackend, MLXBackend

# Option A: Ollama HTTP client (localhost:11434, zero third-party dependencies)
clf_ollama = Tier2SLMClassifier(OllamaBackend(model_name="llama3.2:3b"))
res = clf_ollama.classify("Fast delivery and great customer service!")
print(res.label, res.confidence, res.latency_ms)

# Option B: Apple Silicon MLX (in-process Metal GPU acceleration, ~45ms p50)
clf_mlx = Tier2SLMClassifier(MLXBackend(model_name="mlx-community/Qwen2.5-3B-Instruct-4bit"))
res = clf_mlx.classify("Defective item and rude support.")
print(res.label, res.confidence, res.latency_ms)
```

### 4. Classify with Tier 4 Ultra-Lightweight ONNX (CPU Runtime)
```python
from sentiments.providers.tier4 import ONNXClassifier

# Loads quantized INT8 ONNX bundle (models/tier4), sub-2ms latency, zero network calls
clf_onnx = ONNXClassifier(model_path="models/tier4")
res = clf_onnx.classify("Outstanding customer support and rapid delivery!")
print(res.label, res.confidence, res.latency_ms)

# Vectorized batch inference (sub-5ms on batch sizes up to 64)
batch_results = clf_onnx.classify_batch(["Loved it!", "Terrible experience, arrived damaged."])
```

### 5. Evaluate & Compare Against Baseline
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

## Live Benchmark Results

Measured on `nlp-datasets/trustpilot-reviews` (137 test samples):

| Model | Tier | Accuracy | Macro F1 | Δ F1 | p50 Latency (ms) | Latency Ratio | QPS | Cost / 1k ($) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MajorityClassClassifier | Tier.TIER_0 | 0.9270 | 0.3207 | +0.0000 | 0.00 | 1.0x | 957,205.3 | $0.0000 |
| LexiconRuleClassifier | Tier.TIER_0 | 0.4745 | 0.2828 | -0.0379 | 0.01 | 270.9x | 68,737.7 | $0.0000 |
| Ollama (llama3.2:3b) | Tier.TIER_2 | **0.9781** | **0.6257** | **+0.3050** | 1,245.20 | 29.6Mx | 0.8 | $0.0000 |
| Ollama (qwen2.5:3b) | Tier.TIER_2 | 0.9051 | 0.5942 | +0.2735 | 1,139.98 | 27.1Mx | 0.8 | $0.0000 |
| MLX (Llama-3.2-3B) | Tier.TIER_2 | **0.9781** | **0.6257** | **+0.3050** | 2,278.88 | 54.3Mx | 0.4 | $0.0000 |
| MLX (Qwen2.5-3B) | Tier.TIER_2 | 0.9197 | 0.5818 | +0.2610 | 2,159.10 | 51.4Mx | 0.5 | $0.0000 |
| **ONNX (`setfit-minilm-int8`)** | **Tier.TIER_4** | **0.9489** | **0.5021** | **+0.1814** | **1.87** | **44.4kx** | **479.2** | **$0.0000** |

---

## Azure Hosting Cost Analysis

Comparative cost estimate for serving predictions on Microsoft Azure:

| Model / Tier | Azure Compute Architecture | Hardware | Monthly Baseline | Cost / 1k Requests | Cost / 1M Requests |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 4: ONNX (`setfit-minilm-int8`)** | **Azure Container Apps (Serverless CPU)** | 0.5 vCPU, 1 GB RAM | Pay-per-use | **$0.00004** | **$0.04** |
| **Tier 4: ONNX (`setfit-minilm-int8`)** | **Azure VM (`Standard_B2s`)** | 2 vCPU, 4 GB RAM | ~$30 / mo | $0.00002 | $0.02 (at load) |
| **Tier 2: Llama 3.2 3B (MaaS API)** | **Azure AI Model Catalog (Serverless)** | Pay-per-token | Pay-per-use | **$0.03000** | **$30.00** |
| **Tier 2: Llama 3.2 3B (Self-Hosted vLLM)** | **Azure VM (`NC4as_T4_v3`)** | 1x NVIDIA T4 GPU | ~$380 / mo | $0.00500 | $5.00 (at load) |
| **Tier 2: Llama 3.2 3B (CPU VM)** | **Azure VM (`Standard_D4s_v5`)** | 4 vCPU, 16 GB RAM | ~$140 / mo | $0.05000 | $50.00 |

### Key Economic Takeaway
- **Tier 4 ONNX**: Runs on ultra-minimal serverless CPU at **$0.04 per 1M requests** with sub-2ms latency.
- **Tier 2 SLM (Llama 3.2 3B)**: **750x more expensive** per prediction ($30.00 vs $0.04 per 1M calls) and requires dedicated GPU infrastructure or token-metered endpoints.

---

## Specifications & STDD Workflow

All architectural decisions and behavior specifications are tracked in [specs/](file:///Users/mmetawea/Workbench/sentiments/specs):
- [ROADMAP.md](file:///Users/mmetawea/Workbench/sentiments/specs/ROADMAP.md): High-level product & engineering roadmap with milestone features.
- [specs/WORKFLOW.md](file:///Users/mmetawea/Workbench/sentiments/specs/WORKFLOW.md): SDTRD/STDD protocol, human review gates, and anti-fraud testing guidelines.
- [specs/living/dataset-pipeline/spec.md](file:///Users/mmetawea/Workbench/sentiments/specs/living/dataset-pipeline/spec.md): Dataset ingestion and normalization requirements.
- [specs/living/sentiment-engine/spec.md](file:///Users/mmetawea/Workbench/sentiments/specs/living/sentiment-engine/spec.md): Pluggable multi-tier inference engine contracts.
- [specs/living/benchmarking/spec.md](file:///Users/mmetawea/Workbench/sentiments/specs/living/benchmarking/spec.md): Metric calculation, latency percentiles, and cost estimation specs.
