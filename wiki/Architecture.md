# System Architecture

`sentiments` follows a strict **Hexagonal Architecture (Ports and Adapters)** pattern. The design isolates domain logic, metric evaluations, and data models from external dependencies, model weight runtimes, network daemons, and storage systems.

---

## Hexagonal Core vs I/O Shell

```mermaid
flowchart TD
    subgraph External_Shell["I/O & Infrastructure Shell (Adapters)"]
        CLI["CLI / Benchmark Scripts"]
        DatasetFS["Local Datasets (nlp-datasets/)"]
        OllamaDaemon["Ollama Daemon (:11434)"]
        MetalMLX["Apple Silicon Metal (mlx-lm)"]
        FrontierAPI["Frontier APIs (Claude / OpenAI)"]
        ONNXRuntime["ONNX Runtime Engine"]
    end

    subgraph Ports["Ports & Interfaces"]
        DatasetLoaderPort["DatasetLoader / StratifiedSplitter"]
        ClassifierProtocol["SentimentClassifier Protocol"]
        SLMBackendProtocol["SLMBackend Protocol"]
        MetricsPort["MetricsCalculator / BenchmarkReporter"]
    end

    subgraph Core_Domain["Pure Domain Core"]
        Models["Models: SentimentResult, SentimentLabel, Tier"]
        CoreEval["Core Evaluation: Metrics & Cost Spec"]
        PromptBuilder["Prompt Builder & JSON Output Parser"]
    end

    CLI --> DatasetLoaderPort
    CLI --> ClassifierProtocol
    CLI --> MetricsPort

    DatasetFS --> DatasetLoaderPort
    DatasetLoaderPort --> Models

    ClassifierProtocol --> Models
    MetricsPort --> Models
    MetricsPort --> CoreEval

    OllamaDaemon --> SLMBackendProtocol
    MetalMLX --> SLMBackendProtocol
    SLMBackendProtocol --> PromptBuilder
    PromptBuilder --> Models
    FrontierAPI --> ClassifierProtocol
    ONNXRuntime --> ClassifierProtocol
```

---

## Pure Domain Core

Located under [`src/sentiments/core/`](file:///Users/mmetawea/Workbench/sentiments/src/sentiments/core):
- **[`models.py`](file:///Users/mmetawea/Workbench/sentiments/src/sentiments/core/models.py)**: Immutable value objects:
  - [`SentimentLabel`](file:///Users/mmetawea/Workbench/sentiments/src/sentiments/core/models.py#L6-L9): `POSITIVE`, `NEGATIVE`, `NEUTRAL`.
  - [`Tier`](file:///Users/mmetawea/Workbench/sentiments/src/sentiments/core/models.py#L12-L17): Enumeration of tiers `TIER_0` through `TIER_4`.
  - [`SentimentResult`](file:///Users/mmetawea/Workbench/sentiments/src/sentiments/core/models.py#L20-L36): Frozen dataclass enforcing validation invariants (`confidence` $\in [0, 1]$, `latency_ms` $\ge 0$, non-empty `model_name`).
- **[`classifier.py`](file:///Users/mmetawea/Workbench/sentiments/src/sentiments/core/classifier.py)**: Runtime-checkable [`SentimentClassifier`](file:///Users/mmetawea/Workbench/sentiments/src/sentiments/core/classifier.py#L7-L25) protocol:
  - `tier -> Tier`
  - `model_name -> str`
  - `classify(text: str) -> SentimentResult`
  - `classify_batch(texts: list[str]) -> list[SentimentResult]`

Core domain classes have **zero dependencies** outside the Python standard library.

---

## Provider Architecture & Tier 2 SLM Engine

Tier 2 provides pluggable backends via [`SLMBackend`](file:///Users/mmetawea/Workbench/sentiments/src/sentiments/providers/tier2/protocol.py#L4-L13) protocol:

```mermaid
classDiagram
    class SentimentClassifier {
        <<Protocol>>
        +tier: Tier
        +model_name: str
        +classify(text: str) SentimentResult
        +classify_batch(texts: Sequence[str]) list~SentimentResult~
    }

    class Tier2SLMClassifier {
        +backend: SLMBackend
        +tier: Tier
        +model_name: str
        +classify(text: str) SentimentResult
        +classify_batch(texts: Sequence[str]) list~SentimentResult~
    }

    class SLMBackend {
        <<Protocol>>
        +model_name: str
        +generate(prompt: str, json_schema: bool) str
        +is_available() bool
    }

    class OllamaBackend {
        +base_url: str
        +timeout: float
        +generate(prompt: str, json_schema: bool) str
        +is_available() bool
    }

    class MLXBackend {
        +max_tokens: int
        +generate(prompt: str, json_schema: bool) str
        +is_available() bool
    }

    SentimentClassifier <|.. Tier2SLMClassifier
    Tier2SLMClassifier --> SLMBackend
    SLMBackend <|.. OllamaBackend
    SLMBackend <|.. MLXBackend
```

### Execution Flow: Tier 2 Classification

```mermaid
sequenceDiagram
    autonumber
    actor Caller as Client / Benchmark
    participant Classifier as Tier2SLMClassifier
    participant Core as tier2.core (Prompt/Parser)
    participant Backend as SLMBackend (MLX or Ollama)

    Caller->>Classifier: classify("text")
    Classifier->>Core: build_sentiment_prompt("text")
    Core-->>Classifier: structured prompt with JSON schema
    Classifier->>Backend: generate(prompt, json_schema=True)
    alt MLXBackend
        Backend->>Backend: mlx_lm.generate() (Metal Unified Memory)
    else OllamaBackend
        Backend->>Backend: POST http://localhost:11434/api/generate
    end
    Backend-->>Classifier: raw JSON string
    Classifier->>Core: parse_sentiment_output(raw_output)
    Core-->>Classifier: (SentimentLabel, confidence, metadata)
    Classifier-->>Caller: SentimentResult(label, confidence, latency_ms, tier, model_name)
```
