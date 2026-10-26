# Feature Proposal: Tier 4 Quantized ONNX & SetFit CPU Runtime

## Intent
Deliver an ultra-lightweight, offline CPU sentiment classifier (`Tier.TIER_4`) capable of executing inference in $< 5\text{ms}$ ($p_{50}$) with $> 0.85$ Macro F1 on customer reviews, powered by an INT8 quantized ONNX transformer model and a SetFit contrastive few-shot fine-tuning pipeline.

## Problem Statement
Today, the platform lacks a low-latency, mid-memory, high-accuracy tier:
- **Tier 0** is instant ($< 0.01\text{ms}$) and free, but achieves low Macro F1 ($0.28–0.32$).
- **Tier 2** SLMs reach $0.62$ Macro F1, but require $1,000–2,300\text{ms}$ per request and $2–4\text{GB}$ unified memory.
- There is no production-ready option for high-throughput stream processing, CLI integrations, or edge devices where latency must remain strictly sub-5ms and GPU resources are absent.

## Scope

### In-Scope
- **`ONNXRuntimeBackend`**: Production backend wrapping Microsoft `onnxruntime` and Rust-based `tokenizers`, executing quantized `.onnx` models with dynamic input shape and softmax classification.
- **`ONNXClassifier`**: Production Tier 4 sentiment classifier adhering to `SentimentClassifier` protocol, supporting both single-string (`classify`) and batch (`classify_batch`) inference with sub-5ms budgets and zero network calls.
- **Quantized Model Packaging**: Standalone artifact bundle containing `model_quantized.onnx`, `tokenizer.json`, and class mapping config, loaded without PyTorch or external API connections.
- **SetFit Few-Shot Pipeline**: Offline fine-tuning and export pipeline (`sentiments.training.setfit_pipeline` or CLI runner) that fine-tunes on `StratifiedSplitter` train split (637 samples) and exports directly to INT8 ONNX.
- **Benchmarking & Profiling**: Automated benchmark script evaluating Tier 4 accuracy, Macro F1, $p_{50}/p_{95}$ latency, QPS, and memory footprint against existing baselines.

### Out-of-Scope (YAGNI)
- **GPU / Accelerator Providers**: No CUDA, TensorRT, or CoreML execution providers in runtime; strictly CPU (`CPUExecutionProvider`) for universal zero-config portability.
- **Microservice / Network Server**: No Triton Inference Server, FastAPI, or REST wrapper; strictly in-process Python runtime.
- **Online / Streaming Fine-Tuning**: No dynamic in-flight model weight updating.
- **PyTorch Runtime Dependency**: Production inference runtime (`dependencies`) strictly excludes `torch` and `transformers`; training dependencies are isolated in an optional `train` dependency group.

## Assumptions
- **ASM-GEN-1**: `onnxruntime` is compatible with Python 3.14 on macOS arm64 and Linux.
- **ASM-GEN-2**: Maximum sequence length is bounded (default 128 tokens), comfortably covering 99% of customer reviews without truncation loss.
- **ASM-GEN-3**: Model weights and tokenizers can be packaged locally in the repo or cache directory (`models/tier4/`) for deterministic offline execution.
- **ASM-GEN-4**: Training dependencies (`setfit`, `sentence-transformers`, `torch`) are only invoked during offline model fine-tuning/export, never imported by `ONNXClassifier`.

## Success Metrics
- **METRIC-1**: Inference latency $p_{50} < 5.0\text{ms}$ on standard CPU.
- **METRIC-2**: Macro F1 $> 0.85$ on `nlp-datasets/trustpilot-reviews` test set.
- **METRIC-3**: Resident memory footprint $< 120\text{MB}$ RAM during inference.
- **METRIC-4**: Zero external network calls during classifier lifecycle.
- **METRIC-5**: $\ge 95\%$ test line & branch coverage and 100% pass rate across existing 180 tests.

---

## Review Gate
- [x] **Gate 1 Approval**: Proposal reviewed and approved by human before drafting specs.
