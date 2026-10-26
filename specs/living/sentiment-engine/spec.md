# Specification: Pluggable Sentiment Inference Engine

## 1. Overview
Defines interface contracts and execution semantics for pluggable sentiment classification models spanning Tier 0 (Dumb Baseline) to Tier 4 (Lightweight/ONNX).

## 2. Requirements & Capabilities

### Capability: Inference Interface Contract
- **REQ-ENG-1**: All sentiment classifiers MUST implement the abstract `SentimentClassifier` interface:
  - `classify(text: str) -> SentimentResult`
  - `classify_batch(texts: list[str]) -> list[SentimentResult]`
- **REQ-ENG-2**: `SentimentResult` MUST contain:
  - `label`: Canonical enum (`POSITIVE`, `NEGATIVE`, `NEUTRAL`).
  - `confidence`: Float between `0.0` and `1.0`.
  - `latency_ms`: Float indicating inference execution duration in milliseconds.
  - `tier`: Identifier enum (`TIER_0`, `TIER_1`, `TIER_2`, `TIER_3`, `TIER_4`).
  - `model_name`: String identifier of the active model.

### Capability: Tier 0 Dumb Baseline
- **REQ-ENG-3**: The engine MUST provide a Tier 0 Dumb Baseline implementation:
  - `MajorityClassClassifier`: Predicts the empirical majority class observed during training with confidence equal to class prevalence.
  - `LexiconRuleClassifier`: Evaluates positive vs negative polarity keyword counts (e.g. VADER / Afinn / basic polarity dictionary) with zero GPU overhead.
- **REQ-ENG-4**: Tier 0 inference MUST execute with average latency < 1.0 ms per record and zero external network calls.

### Capability: Higher Tier Providers (Tier 1 to 4)
- **REQ-ENG-5**: Tier 1 (Frontier API Provider): Configurable client interfacing with LLM endpoints (prompt template + response parsing + token usage tracking).
- **REQ-ENG-6**: Tier 2 (Distilled / Compact SLM Provider): Unified `Tier2SLMClassifier` supporting local backends (`OllamaBackend` via HTTP `localhost:11434` and `MLXBackend` via Apple Silicon Metal acceleration) for compact models (e.g. `llama3.2:3b`, `qwen2.5:3b`) with deterministic JSON schema extraction, markdown fence stripping, and confidence normalization.
- **REQ-ENG-7**: Tier 3 (Fine-Tuned Encoder Provider): Local transformer encoder (e.g., PyTorch/HuggingFace RoBERTa/DeBERTa) pipeline.
- **REQ-ENG-8**: Tier 4 (Ultra-Lightweight ONNX Provider): Production `ONNXClassifier` backed by `ONNXRuntimeBackend` using `onnxruntime` (`CPUExecutionProvider`) and Rust `tokenizers.Tokenizer`. Executes pre-quantized INT8 sentence transformer models (e.g. SetFit MiniLM) with median latency $p_{50} < 5.0\text{ms}$, throughput $\ge 150\text{ QPS}$, $< 120\text{MB}$ RAM, and zero runtime network calls. Resolves local model bundles containing `model_quantized.onnx`, `tokenizer.json`, and `config.json`. Supports resilient batch fallback when `fail_fast=False`.

## 3. Assumptions
- **ASM-ENG-1**: Taxonomy is strictly 3-class (`POSITIVE`, `NEGATIVE`, `NEUTRAL`) across all tiers for parity.
- **ASM-ENG-2**: Python 3.10+ execution environment.
- **ASM-ENG-3**: Tier 0 runs completely offline on standard CPU without dependencies on external binaries or network.
- **ASM-ENG-4**: Tier 1 providers require user-supplied API keys via environment variables; mock providers used for automated test suites.
- **ASM-ENG-5**: Latency is measured as wall-clock elapsed time in milliseconds for the inference step.

## 4. Edge Cases & Error Handling
- **Empty or Whitespace Input**: Raising `InvalidInputError` or returning default `NEUTRAL` with 0 confidence based on configuration.
- **Exceeded Context Length**: Gracefully truncating or handling oversized inputs without crashing.
- **Provider API Outage**: Catching network/timeout exceptions in Tier 1 and wrapping into structured `ClassifierProviderError`.
