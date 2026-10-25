# Feature Proposal: Tier 2 Local SLM Classifier (Ollama & MLX)

## 1. Intent
Deliver local, production-grade Small Language Model (SLM) sentiment classification for `sentiments`, executing on developer laptops and Apple Silicon without cloud API fees or latency penalties.

## 2. Problem Statement
Tier 0 baselines (`MajorityClassClassifier`, `LexiconRuleClassifier`) achieve at most 0.3207 Macro F1 and fail to understand conversational nuances or negative feedback. Tier 1 frontier models (OpenAI, Anthropic) incur per-token financial costs and high external network latency. Developers need a self-contained Tier 2 classification engine capable of high accuracy ($\text{Macro F1} > 0.70$) running locally with low latency ($\le 100\text{ms}$).

## 3. Scope

### In-Scope (Ponytail / YAGNI)
- **Model Support**: Zero-shot and few-shot classification targeting `llama3.2:3b` and `qwen2.5:3b` (and compatible variants).
- **Dual Backends**:
  1. `OllamaBackend`: Connects to local Ollama HTTP REST API (`localhost:11434`) via Python standard library `urllib`.
  2. `MLXBackend`: In-process Apple Silicon Metal acceleration via dynamic integration with `mlx-lm`.
- **Structured Schema & Normalization**: Strict prompt template enforcing valid JSON output (`label`, `confidence`, token telemetry), mapping directly into `SentimentResult`.
- **Backend Protocol & Fallback**: Unified `Tier2SLMClassifier` interface allowing runtime backend selection with clear diagnostics when daemons/dependencies are missing.

### Out-of-Scope (YAGNI)
- Model fine-tuning / LoRA training (deferred to Milestone 5 / `feat/05`).
- Multi-node distributed inference or cluster orchestration.
- Non-sentiment downstream generative tasks (chat, summarization).
- Non-Metal GPU backends (e.g. ROCm, DirectML).

## 4. Success Metrics
- **Classification Quality**: Macro F1 $\ge 0.70$ on held-out review test data (surpassing Tier 0 baseline of 0.3207).
- **Runtime Latency**: $p_{50} \le 100\text{ms}$ per review on Apple Silicon M-series.
- **Cost**: $0.00 marginal inference cost per 1k requests.
- **Reliability**: $0\%$ crashes on malformed model responses; graceful fallback to `NEUTRAL` or diagnostic error.
- **Test Integrity**: $\ge 95\%$ line & branch coverage; $100\%$ mutation kill score on core logic.

## 5. Assumptions
- **ASM-GEN-1**: Users wishing to run the Ollama backend have the Ollama CLI installed and running (`ollama serve`).
- **ASM-GEN-2**: Users wishing to run the MLX backend operate on macOS with Apple Silicon (Metal) and install `mlx-lm`.
- **ASM-GEN-3**: Supported models accept conversational chat templates or system/user instruction prompt formats.
- **ASM-GEN-4**: Prompt tokens generated adhere to standard tri-class sentiment values (`POSITIVE`, `NEGATIVE`, `NEUTRAL`).
