# Specification: Tier 4 Quantized ONNX & SetFit CPU Runtime

## Capability: `tier4-onnx-runtime`

### 1. Functional Requirements

- **REQ-T4-1 (Classifier Interface Compliance)**: The system MUST provide `ONNXClassifier` implementing the `SentimentClassifier` protocol:
  - `classify(text: str) -> SentimentResult`
  - `classify_batch(texts: Sequence[str], fail_fast: bool = False) -> list[SentimentResult]`
- **REQ-T4-2 (Tier & Model Attribution)**: Every `SentimentResult` returned by `ONNXClassifier` MUST set `tier = Tier.TIER_4` and `model_name` matching the model identifier (e.g., `minilm-int8` or the artifact basename).
- **REQ-T4-3 (Backend Abstraction)**: `ONNXClassifier` MUST operate through an `ONNXBackend` protocol:
  - `predict(text: str) -> tuple[SentimentLabel, float, dict[str, Any]]`
  - `predict_batch(texts: Sequence[str]) -> list[tuple[SentimentLabel, float, dict[str, Any]]]`
  - `is_available() -> bool`
  - `model_name: str`
- **REQ-T4-4 (ONNXRuntime Engine Execution)**: `ONNXRuntimeBackend` MUST execute inference using `onnxruntime.InferenceSession` configured with `CPUExecutionProvider` and HuggingFace Rust `tokenizers.Tokenizer` for local tokenization.
- **REQ-T4-5 (Logits & Softmax Normalization)**: When the ONNX model outputs unnormalized logits, `ONNXRuntimeBackend` MUST compute numerically stable softmax, select the argmax class, map it to `SentimentLabel` (`POSITIVE`, `NEGATIVE`, `NEUTRAL`), and clamp confidence to $[0.0, 1.0]$.
- **REQ-T4-6 (Zero Network Execution)**: The runtime engine MUST execute 100% offline. No external HTTP requests, sockets, or telemetry are permitted during model initialization or inference.
- **REQ-T4-7 (SetFit Artifact Support)**: The backend MUST support packaged SetFit artifacts containing an ONNX sentence transformer encoder paired with a serialized linear head, or an end-to-end fused ONNX graph.
- **REQ-T4-8 (Batched Vectorization)**: `predict_batch` in `ONNXRuntimeBackend` MUST perform batched tokenization and a single vectorized ONNX session invocation when processing multiple texts.

---

### 2. Non-Functional Requirements & Budgets

- **NFR-T4-1 (Latency Budget)**: Single-sample inference latency on CPU MUST satisfy:
  - Median ($p_{50}$) $< 5.0\text{ms}$.
  - 95th percentile ($p_{95}$) $< 10.0\text{ms}$.
- **NFR-T4-2 (Throughput Budget)**: Vectorized batch inference on CPU (batch size 64) MUST achieve throughput $\ge 150\text{ QPS}$.
- **NFR-T4-3 (Memory Footprint Budget)**: Peak resident memory (RSS) consumed by the ONNX session and tokenizer MUST be $< 120\text{MB}$.
- **NFR-T4-4 (Zero PyTorch Runtime Dependency)**: The inference runtime module (`src/sentiments/providers/tier4/`) MUST NOT import `torch`, `transformers`, or `accelerate`.

---

### 3. Assumptions & Edge Cases

- **ASM-T4-1**: Model artifacts are stored locally in a directory containing `model.onnx` (or `model_quantized.onnx`), `tokenizer.json`, and `config.json`.
- **ASM-T4-2**: The tokenizer truncates inputs to a maximum sequence length (default 128 tokens) with padding enabled.
- **Edge Case 1 (Empty or Whitespace Input)**: Input string that is empty `""` or contains only whitespace MUST raise `ValueError("text must be a non-empty string")`.
- **Edge Case 2 (Oversized Input)**: Inputs exceeding max sequence length MUST be deterministically truncated to max length without error.
- **Edge Case 3 (Missing or Corrupted Artifact)**: If the specified model file or tokenizer does not exist or fails to load, `ONNXRuntimeBackend` MUST raise a structured `ClassifierProviderError`.
- **Edge Case 4 (Batch Failure Resilience)**: In `classify_batch`, if `fail_fast=False` and an individual sample fails, the engine MUST return a fallback `SentimentResult` (`NEUTRAL`, confidence `0.33`, latency `0.0`, metadata with error details) without aborting the batch.

---

### 4. Behavioral Scenarios (Given / When / Then)

#### Scenario 4.1: Single review sentiment classification via ONNX
- **Given** an initialized `ONNXClassifier` with a valid INT8 ONNX model
- **When** `classify("Great product, loved it!")` is called
- **Then** the result label MUST be `SentimentLabel.POSITIVE`
- **And** confidence MUST be $> 0.5$ and $\le 1.0$
- **And** `tier` MUST be `Tier.TIER_4`
- **And** latency MUST be recorded and $> 0.0\text{ms}$

#### Scenario 4.2: Vectorized batch classification
- **Given** an initialized `ONNXClassifier` with 5 sample customer reviews
- **When** `classify_batch(texts)` is called
- **Then** exactly 5 `SentimentResult` objects MUST be returned in order
- **And** each result MUST have a valid `SentimentLabel` and non-negative latency

#### Scenario 4.3: Empty or whitespace input rejection
- **Given** an initialized `ONNXClassifier`
- **When** `classify("   ")` is called
- **Then** a `ValueError` MUST be raised with message containing `"text must be a non-empty string"`

#### Scenario 4.4: Truncation of oversized text
- **Given** an initialized `ONNXClassifier`
- **When** `classify(text)` is called with a 1,000-word review
- **Then** the review MUST be processed successfully without error
- **And** a valid `SentimentResult` MUST be returned

#### Scenario 4.5: Missing model artifact raises ClassifierProviderError
- **Given** a path to a non-existent ONNX model directory
- **When** `ONNXRuntimeBackend` is initialized
- **Then** a `ClassifierProviderError` MUST be raised detailing the missing artifact

---

## Review Gate
- [x] **Gate 2 Approval**: Specifications reviewed and approved by human before technical design.
