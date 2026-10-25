# Specification: Tier 2 Local SLM Classifier (Ollama & MLX)

## 1. Overview
Defines requirements, runtime contracts, and behavioral specifications for local Small Language Model (SLM) sentiment classification across Ollama and Apple MLX execution backends.

---

## 2. Requirements & Capabilities

### Capability: Backend Protocol & Provider Core
- **REQ-SLM-1**: The system MUST implement a unified `Tier2SLMClassifier` satisfying the core classifier protocol with `tier = Tier.TIER_2`.
- **REQ-SLM-2**: The system MUST define an abstract backend interface (`SLMBackend`) with `generate(prompt: str, json_schema: bool) -> str` and `is_available() -> bool`.
- **REQ-SLM-3**: `OllamaBackend` MUST communicate with an Ollama HTTP server (default `http://localhost:11434`) using standard library HTTP client without external dependencies.
- **REQ-SLM-4**: `MLXBackend` MUST interface with Apple MLX (`mlx-lm`) for in-process Metal inference on Apple Silicon, dynamically checking runtime availability.
- **REQ-SLM-5**: Supported default model identifiers MUST include `llama3.2:3b` and `qwen2.5:3b`.

### Capability: Prompting & Output Normalization
- **REQ-SLM-6**: The classifier MUST format review text into a deterministic system/user prompt instructing the model to output valid JSON: `{"label": "<POSITIVE|NEGATIVE|NEUTRAL>", "confidence": <float>, "explanation": "<str>"}`.
- **REQ-SLM-7**: The parser MUST robustly extract JSON from model output, stripping Markdown code fences (````json ... ````) or extraneous conversational tokens.
- **REQ-SLM-8**: Extracted labels MUST be validated against `SentimentLabel` enum (`POSITIVE`, `NEGATIVE`, `NEUTRAL`). Invalid or unparseable labels MUST raise or fallback deterministically without unhandled crashes.
- **REQ-SLM-9**: Extracted confidence scores MUST be clamped between `0.0` and `1.0`. If omitted by model, default to `1.0`.

### Capability: Performance & Non-Functional Budgets
- **NFR-SLM-1**: P50 latency for local classification on Apple Silicon M-series MUST NOT exceed `100ms` per sample for models $\le 3\text{B}$ parameters.
- **NFR-SLM-2**: Marginal cost per 1,000 predictions MUST be reported as `$0.00` in benchmark telemetry.
- **NFR-SLM-3**: Peak memory footprint for 4-bit 3B models MUST NOT exceed `3.5GB` RAM.
- **NFR-SLM-4**: Code test coverage MUST achieve $\ge 95\%$ line and branch coverage; mutation score MUST achieve $100\%$ kill rate on core logic.

---

## 3. Assumptions
- **ASM-SLM-1**: The Ollama daemon is reachable over TCP loopback (`127.0.0.1` / `localhost`).
- **ASM-SLM-2**: MLX backend execution occurs on macOS 13.0+ with Apple Silicon (Metal).
- **ASM-SLM-3**: The target SLM produces coherent classification responses when prompted with temperature = 0.0 or low temperature.

---

## 4. Edge Cases & Error Handling
- **Connection Refusal**: When Ollama daemon is offline, `OllamaBackend` MUST raise `ClassifierProviderError` with descriptive actionable diagnosis.
- **Model Not Found**: When requested model is not downloaded on Ollama/MLX, MUST raise `ClassifierProviderError` indicating missing model name.
- **Malformed JSON Output**: When model produces non-JSON or partial response, the parser MUST attempt regex extraction; if unrecoverable, raise `ClassifierProviderError` or fall back to safe neutral prediction.
- **Empty or Whitespace Input**: Input validation MUST reject empty/invalid strings with `ValueError` consistent with existing classifier contracts.
- **MLX Missing on Host**: If `mlx-lm` is not installed or running on non-Apple platform, `MLXBackend.is_available()` MUST return `False` and instantiation MUST raise `ClassifierProviderError`.

---

## 5. Behavioral Scenarios

### Scenario 1: Successful classification via Ollama backend
- **Given** an active Ollama backend running `llama3.2:3b`
- **When** classifying review `"The product quality exceeded all my expectations!"`
- **Then** the result label is `SentimentLabel.POSITIVE`
- **And** confidence is between `0.0` and `1.0`
- **And** tier is `Tier.TIER_2`
- **And** latency is recorded in milliseconds.

### Scenario 2: Successful classification via MLX backend
- **Given** an available MLX backend running `qwen2.5:3b`
- **When** classifying review `"Completely defective item and rude customer support."`
- **Then** the result label is `SentimentLabel.NEGATIVE`
- **And** tier is `Tier.TIER_2`.

### Scenario 3: Graceful error handling when daemon is unreachable
- **Given** an Ollama backend configured to an unreachable port
- **When** classifying any text
- **Then** a `ClassifierProviderError` is raised with connection failure details.

### Scenario 4: Handling markdown-wrapped JSON output
- **Given** a backend returning ```json\n{"label": "NEUTRAL", "confidence": 0.8}\n```
- **When** parsing the response
- **Then** the parser extracts label `SentimentLabel.NEUTRAL` and confidence `0.8`.
