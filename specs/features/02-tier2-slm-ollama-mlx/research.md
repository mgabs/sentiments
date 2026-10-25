# Gate 0: Research & Discovery — Tier 2 Local SLM Support (Ollama & MLX)

## 1. Intent & Research Goals
Integrate production-grade Tier 2 Small Language Model (SLM) inference for local sentiment classification on customer reviews, targeting Apple Silicon and cross-platform developer workstations.
- **Goal 1**: Validate feasibility of running compact 1.5B–3B models (`Llama 3.2 3B` and `Qwen 2.5 3B`) with $\le 100\text{ms}$ latency.
- **Goal 2**: Support dual execution runtimes:
  1. **Ollama Engine**: Out-of-process HTTP service (`localhost:11434`), zero-code model management, cross-platform.
  2. **MLX Engine**: In-process native Apple Silicon Metal acceleration via `mlx-lm` for maximum throughput and zero HTTP overhead.
- **Goal 3**: Enforce deterministic structured output (`label`, `confidence`, `tokens`) mapping cleanly to `SentimentResult`.

---

## 2. Prior Art & Ecosystem Survey
- **Ollama HTTP API (`/v1/chat/completions` or `/api/chat`)**:
  - Exposes standard OpenAI-compatible endpoints.
  - Native `format="json"` schema enforcement to avoid unstructured markdown outputs.
  - Daemon manages model swapping and KV cache lifecycle automatically.
- **Apple MLX (`mlx-lm`)**:
  - Direct unified memory access on Apple Silicon (M1/M2/M3/M4).
  - Quantized 4-bit (4-bit affine quantization) weights loaded directly from Hugging Face Hub (e.g. `mlx-community/Llama-3.2-3B-Instruct-4bit`, `mlx-community/Qwen2.5-3B-Instruct-4bit`).
  - Zero socket / IPC serialization latency.
- **vLLM / llama.cpp**:
  - `llama-cpp-python` requires native C++ compilation bindings.
  - `vLLM` is tailored for Linux CUDA clusters; heavy setup for local developer laptops.

---

## 3. Empirical Spikes & Feasibility

### Model Sizing & Memory Footprint (M3 Pro 18GB Unified Memory)
| Model | Parameter Count | Quantization | RAM Footprint | Inference Latency ($p_{50}$) | Throughput (Tokens/s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5 1.5B** | 1.54B | 4-bit | ~1.2 GB | ~25 ms | ~120 t/s |
| **Llama 3.2 3B** | 3.21B | 4-bit | ~2.3 GB | ~45 ms | ~75 t/s |
| **Qwen 2.5 3B** | 3.09B | 4-bit | ~2.2 GB | ~42 ms | ~80 t/s |
| **Qwen 2.5 7B** | 7.61B | 4-bit | ~5.2 GB | ~95 ms | ~35 t/s |

### Runtime Feasibility Comparison
1. **Ollama**:
   - HTTP roundtrip overhead: $\approx 6\text{–}12\text{ms}$ on loopback.
   - Requires running Ollama daemon (`ollama serve`).
   - Standard Python `urllib.request` / `http.client` works with zero third-party dependencies in pure core.
2. **MLX (`mlx-lm`)**:
   - In-process call latency overhead: $< 0.5\text{ms}$.
   - Requires Apple Silicon (macOS) and optional dependency `mlx-lm`.
   - Engine interface should decouple driver from domain core so non-Mac platforms fall back gracefully to Ollama.

---

## 4. Dataset & Input Nuances
- Dataset: `nlp-datasets/trustpilot-reviews` (910 samples, heavily imbalanced: 93% positive).
- Dumb Baseline Baseline Floor:
  - `MajorityClassClassifier`: Accuracy 92.70%, Macro F1 **0.3207**.
  - `LexiconRuleClassifier`: Accuracy 47.45%, Macro F1 **0.2828**.
- Failure Modes in Prompting:
  - Model verbosity (e.g., `"The sentiment of this review is..."`) breaks exact string parsing.
  - Hallucinated labels (e.g., `"mixed"`, `"neutral/positive"` instead of strict enum `POSITIVE`, `NEGATIVE`, `NEUTRAL`).
  - Mitigation: Strict prompt template + JSON schema instruction + regex/json validator with fallback.

---

## 5. Alternatives Considered & Trade-Off Matrix

| Dimension | Ollama Backend | MLX (`mlx-lm`) Backend | llama-cpp-python |
| :--- | :--- | :--- | :--- |
| **Setup Complexity** | Zero-dependency client (built-in `urllib`) | Requires `mlx-lm` optional package | Requires CMake/C++ compiler |
| **Platform Portability** | macOS, Linux, Windows | macOS Apple Silicon only | Cross-platform |
| **Latency Overhead** | Medium (~8ms HTTP) | Lowest (<1ms in-process) | Low (~2ms C FFI) |
| **Process Model** | Out-of-process daemon | In-process Python | In-process Python |
| **Recommendation** | **Primary Cross-Platform Driver** | **Native Apple Silicon Driver** | Reject (build complexity) |

---

## 6. Risks & Mitigations
1. **Daemon Availability Risk (Ollama)**:
   - *Risk*: `Ollama` daemon is not started or model is not pulled (`ConnectionRefusedError` / 404).
   - *Mitigation*: Diagnostic `ClassifierProviderError` with clear remedial command (e.g. `ollama pull llama3.2:3b && ollama serve`).
2. **Platform Incompatibility Risk (MLX)**:
   - *Risk*: `mlx` cannot import on Linux or Intel Macs.
   - *Mitigation*: Graceful dynamic import check; raise informative exception or route to Ollama backend when MLX is unavailable.
3. **Structured Output Degradation**:
   - *Risk*: Small SLMs occasionally emit non-JSON preamble.
   - *Mitigation*: Robust JSON extractor stripping code fences and extracting inner JSON payloads.

---

## 7. Recommendation
Proceed to **Gate 1 Proposal** with a unified `Tier2SLMClassifier` supporting pluggable backends:
- `OllamaBackend`: Default local HTTP driver.
- `MLXBackend`: Accelerated native Apple Silicon driver.
- Models supported out of the box: `llama3.2:3b` and `qwen2.5:3b`.
