# Technical Design: Sentiment Analysis Engine & Multi-Tier Benchmarking

## 1. Architecture Overview

The system is decoupled into four modular domains:
1. **Dataset Domain (`sentiment.dataset`)**: Discovers, ingests, normalizes, and deterministically splits corpora from `nlp-datasets/`.
2. **Core Domain (`sentiment.core`)**: Pure domain models, canonical taxonomies, and abstract classifier contracts.
3. **Provider Domain (`sentiment.providers`)**: Concrete classifier implementations across Tier 0 (Dumb Baseline) through Tier 4.
4. **Evaluation Domain (`sentiment.eval`)**: Metrics computation, latency profiling, token/compute cost estimation, and comparative reporting.

```mermaid
flowchart TD
    subgraph Data Layer
        A["nlp-datasets/"] --> B[Dataset Loader & Normalizer]
        B --> C[Stratified Splitter]
        C --> D[(Train / Val / Test Splits)]
    end

    subgraph Classifier Tier Providers
        E[SentimentClassifier Interface]
        E --> T0["Tier 0: Dumb Baseline (Lexicon / Majority)"]
        E --> T1["Tier 1: Frontier LLM (Few-Shot Prompting)"]
        E --> T2["Tier 2: Compact SLM (vLLM / Serverless)"]
        E --> T3["Tier 3: Fine-Tuned Encoder (RoBERTa/DeBERTa)"]
        E --> T4["Tier 4: Ultra-Lightweight (Quantized ONNX)"]
    end

    subgraph Benchmarking & Reporting
        D -. Test Set .-> G[Benchmark Runner]
        T0 & T1 & T2 & T3 & T4 --> G
        G --> H[Metrics Engine (F1, Accuracy, Latency, Cost)]
        H --> I[Markdown / JSON Tier Comparison Table]
    end
```

---

## 2. Component Design & Contracts

### 2.1 Domain Models & Interfaces (`sentiment.core`)

```python
from enum import Enum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

class Tier(str, Enum):
    TIER_0 = "TIER_0"  # Dumb Baseline
    TIER_1 = "TIER_1"  # Frontier LLM
    TIER_2 = "TIER_2"  # Compact SLM / MoE
    TIER_3 = "TIER_3"  # Fine-Tuned Encoder
    TIER_4 = "TIER_4"  # Ultra-Lightweight / ONNX

@dataclass(frozen=True)
class SentimentResult:
    label: SentimentLabel
    confidence: float
    latency_ms: float
    tier: Tier
    model_name: str
    metadata: dict[str, any]

@runtime_checkable
class SentimentClassifier(Protocol):
    @property
    def tier(self) -> Tier: ...
    @property
    def model_name(self) -> str: ...
    def classify(self, text: str) -> SentimentResult: ...
    def classify_batch(self, texts: list[str]) -> list[SentimentResult]: ...
```

### 2.2 Dataset Pipeline (`sentiment.dataset`)
- **`DatasetLoader`**: Scans `nlp-datasets/`, inspects CSV delimiters (`,` vs `;`), detects target text columns (`Content`, `Comments`, `Reviews`) and target label columns (`Tag`, `Consumer Sentiment Classifier`).
- **`LabelNormalizer`**: Maps heterogeneous tags (e.g. `positive`, `negative`, `Pos`, `Neg`) to canonical `SentimentLabel`. Unresolvable tags are safely excluded.
- **`DatasetSplitter`**: Performs deterministic stratified random splitting (e.g. 70/15/15) with fixed seeds to preserve class distributions.

### 2.3 Tier 0 Dumb Baseline (`sentiment.providers.tier0`)
- **`MajorityClassClassifier`**: Fits on training split, caches the mode class, outputs class prevalence as confidence. Execution latency <0.01ms.
- **`LexiconRuleClassifier`**: Fast word-tokenization; matches against positive/negative sentiment lexicons; computes net polarity ratio. Execution latency <0.5ms.

### 2.4 Extensible Tiers 1–4 (`sentiment.providers`)
- **Tier 1 (`tier1.FrontierLLMClassifier`)**: Handles prompt formatting, client calls, JSON response validation, and token usage accounting.
- **Tier 2 (`tier2.CompactSLMClassifier`)**: Interfaces with local/hosted OpenAI-compatible inference endpoints (vLLM / Ollama).
- **Tier 3 (`tier3.FineTunedEncoderClassifier`)**: Wraps Hugging Face transformers pipeline with truncation/batching.
- **Tier 4 (`tier4.ONNXClassifier`)**: Loads optimized `.onnx` models executing via ONNX Runtime CPU.

### 2.5 Benchmarking Engine (`sentiment.eval`)
- Computes:
  - Classification metrics: Accuracy, Macro F1, Macro Precision, Macro Recall, per-class Confusion Matrix.
  - Efficiency metrics: Latency percentiles ($p_{50}, p_{95}, p_{99}$), Throughput ($QPS$).
  - Economics: Estimated cost per 1,000 requests based on model token rates.
  - Baseline Comparative Delta:
    $$\Delta F1 = F1_{Tier\_N} - F1_{Tier\_0}$$
    $$\text{Latency Ratio} = \frac{\text{Latency}_{Tier\_0}}{\text{Latency}_{Tier\_N}}$$

---

## 3. Tech Stack & Dependencies

- **Package & Environment Management**: `uv`
- **Language**: Python 3.10+
- **Testing & Quality Assurance**:
  - `pytest`: Core unit & integration runner.
  - `pytest-bdd` / `behave`: BDD feature specification and scenario execution (Target: 100% behavior coverage).
  - `pytest-cov`: Code coverage monitoring (Target: >= 95%).
  - `mutmut`: Mutation testing framework (Target: 100% mutant kill rate on core logic).
- **Data & Evaluation**: Standard library (`csv`, `math`, `time`, `re`, `collections`, `dataclasses`) for core & baseline to avoid unnecessary dependencies (YAGNI).

---

## 4. Review Gate
- [x] **Gate 3 Approval**: Manual user sign-off approved.
