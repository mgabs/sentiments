# Local SLM Developer Guide (Tier 2)

This guide covers running 3B small language models (`llama3.2:3b` and `qwen2.5:3b`) locally using either native Apple Silicon Metal acceleration via **MLX** or cross-platform HTTP execution via **Ollama**.

---

## 1. Prerequisites & Installation

### Option A: MLX Native (macOS Apple Silicon)
MLX compiles directly to Apple Silicon unified memory with Metal acceleration, eliminating HTTP loopback latency.

```bash
# Install MLX LM dependency inside the virtual environment
uv pip install mlx-lm
```

Recommended 4-bit models on HuggingFace:
- `mlx-community/Qwen2.5-3B-Instruct-4bit` (Default)
- `mlx-community/Llama-3.2-3B-Instruct-4bit`

### Option B: Ollama Daemon (macOS, Linux, Windows)
Ollama runs an isolated inference server exposing an HTTP API on port `11434`.

```bash
# 1. Install Ollama via Homebrew (or download from ollama.com)
brew install ollama

# 2. Start the background service
ollama serve

# 3. Pull target 3B models
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
```

---

## 2. Using Tier2SLMClassifier in Code

### Quickstart with MLXBackend (Recommended for Mac)
```python
from sentiments.providers.tier2 import Tier2SLMClassifier, MLXBackend

# Initialized with default 4-bit Qwen2.5-3B
mlx_backend = MLXBackend(model_name="mlx-community/Qwen2.5-3B-Instruct-4bit")
classifier = Tier2SLMClassifier(backend=mlx_backend)

result = classifier.classify("The handmade mug arrived cracked, but the seller refunded me instantly.")

print(f"Sentiment : {result.label.value}")       # POSITIVE, NEGATIVE, or NEUTRAL
print(f"Confidence: {result.confidence:.2f}")     # 0.0 - 1.0
print(f"Latency   : {result.latency_ms:.1f} ms")  # ~45 ms on M-series
print(f"Metadata  : {result.metadata}")          # {'aspect': '...', 'reasoning': '...'}
```

### Quickstart with OllamaBackend
```python
from sentiments.providers.tier2 import Tier2SLMClassifier, OllamaBackend

# Uses local Ollama HTTP daemon
ollama_backend = OllamaBackend(model_name="llama3.2:3b")
classifier = Tier2SLMClassifier(backend=ollama_backend)

result = classifier.classify("Shipping took three weeks and packaging was damaged.")
print(result.label, result.confidence, result.latency_ms)
```

---

## 3. Structured Output & Schema Enforcement

Tier 2 classifiers enforce strict JSON generation using [`build_sentiment_prompt()`](file:///Users/mmetawea/Workbench/sentiments/src/sentiments/providers/tier2/core.py#L9-L28) and parse responses with [`parse_sentiment_output()`](file:///Users/mmetawea/Workbench/sentiments/src/sentiments/providers/tier2/core.py#L31-L71).

### Expected Output Schema:
```json
{
  "label": "POSITIVE | NEGATIVE | NEUTRAL",
  "confidence": 0.95,
  "aspect": "customer service",
  "reasoning": "Prompt refund resolved the product defect"
}
```

The parser gracefully handles:
- Markdown fenced code blocks (` ```json ... ``` `).
- Unescaped raw string responses.
- Missing optional metadata fields while ensuring confidence is clamped to `[0.0, 1.0]`.

---

## 4. Batch Inference & Performance Tuning

For evaluating batches of reviews:

```python
texts = [
    "Five stars! Outstanding quality.",
    "Completely useless, broke on day one.",
    "Standard delivery, average quality.",
]

results = classifier.classify_batch(texts)
for res in results:
    print(f"[{res.label.value}] ({res.latency_ms:.1f}ms): {res.metadata.get('reasoning', '')}")
```

### Performance Tuning Tips:
- **MLX Warm-Up**: The first inference call triggers weight compilation into unified memory. Warm up the model with a dummy request before benchmarking.
- **Token Limits**: Keep `max_tokens` set to `128` (default). Structured sentiment JSON responses require less than 60 tokens; lower limits prevent generation runaway.
- **Ollama Timeout**: Increase `timeout` in [`OllamaBackend`](file:///Users/mmetawea/Workbench/sentiments/src/sentiments/providers/tier2/ollama.py#L11-L19) if running under heavy concurrency or on older hardware.
