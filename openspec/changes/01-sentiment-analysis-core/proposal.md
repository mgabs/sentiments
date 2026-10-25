# Proposal: Sentiment Analysis Engine & Multi-Tier Benchmarking Framework

## 1. Intent & Problem Statement
Build an end-to-end sentiment analysis application using customer feedback data from `nlp-datasets/`. The system must ingest labeled training data from the repository, establish a clear dumb baseline, and provide an extensible benchmarking framework comparing all tiers across accuracy, cost, latency, throughput, and frontier model dependency.

Follow strict Spec- and Test-Driven Development (STDD): 100% BDD behavior coverage, >= 95% unit test coverage, 100% mutation test score.

## 2. Model Evaluation & Comparison Matrix

| Approach / Tier | Dependency on Frontier LLMs | Operational Cost | Latency & Throughput Efficiency | Best Suited For |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 0: Dumb Baseline**<br>*(e.g., Lexicon/VADER, Majority Class, TF-IDF + Logistic Regression)* | **Zero**<br>Trivial heuristics or basic statistical counting. | **Zero**<br>Runs anywhere on minimal CPU cycles. | **Instantaneous**<br><1ms latency; tens of thousands of queries/sec. | Floor baseline to prove that complex/expensive tiers justify their compute cost. |
| **Tier 1: Frontier Zero/Few-Shot Reasoning**<br>*(e.g., Claude 3.5/Opus, GPT-4o)* | **Direct / Absolute**<br>Relies entirely on proprietary frontier API endpoints. | **High to Extreme**<br>~$0.005–$0.03 per 1K calls; large context overhead for few-shot prompt templates. | **Poor / High Latency**<br>500ms–2500ms TTFT; low throughput without costly concurrent API provisioning. | Multi-hop emotion cause extraction (ECPE), implicit sarcasm, unstandardized taxonomy synthesis, and offline edge-case audits. |
| **Tier 2: Distilled / Compact SLMs & Small MoEs**<br>*(e.g., Qwen 3B–8B, Llama 8B, Phi-4)* | **Indirect / Transitional**<br>Uses frontier LLMs offline as "teacher" models to generate synthetic labels or LoRA training pairs. | **Moderate to Low**<br>Hosted on cheap GPU/vLLM instances or budget serverless APIs (~$0.05–$0.20 / 1M tokens). | **Moderate to High**<br>50ms–150ms per call; batchable via TensorRT-LLM or vLLM. | Structured Aspect-Based Sentiment Analysis (ABSA), multi-lingual product feedback, and medium-volume conversational triage. |
| **Tier 3: Specialized Fine-Tuned Encoders**<br>*(e.g., RoBERTa-large, DeBERTa-v3)* | **Zero**<br>Self-contained bidirectional encoder; no generative LLM calls during training or inference. | **Minimal**<br>Commodity single-GPU or shared multi-core cloud instances; fixed infra cost. | **Very High**<br>10ms–40ms per batch; high saturation on low-end accelerators. | Standard document/sentence polarity classification, customer support routing, and high-frequency stream monitoring. |
| **Tier 4: Ultra-Lightweight & Distilled Embeddings**<br>*(e.g., DistilBERT, SetFit, Quantized ONNX)* | **Zero**<br>Completely independent; often runs on CPU or client edge devices. | **Near Zero**<br>Runs on standard CPUs without GPU overhead; negligible cloud egress. | **Ultra-High / Real-Time**<br><5ms latency; thousands of queries/sec per CPU node. | High-throughput firehoses (real-time ad-tech, clickstream logs, live gaming chat, social media feeds). |

## 3. Scope

### In-Scope
- **Dumb Baseline (Tier 0)**: Immediate heuristic/statistical benchmark (lexicon rule-based / majority class / simple bag-of-words) establishing minimum acceptable performance floor.
- **Dataset Ingestion & Splitter**: Parse and normalize labeled datasets from `nlp-datasets/` (`training-sets/`, labeled subsets of `trustpilot-reviews/`, `udemy-reviews/`, `etsy-reviews/`) into standardized train/val/test splits.
- **Pluggable Sentiment Engine Architecture**: Unified interface (`classify(text) -> SentimentResult`) supporting model tier providers (Tier 0 through Tier 4).
- **Benchmarking & Comparison Pipeline**: Automated evaluation harness measuring:
  - Quality: Accuracy, Macro F1, Precision, Recall, Confusion Matrix relative to Dumb Baseline.
  - Performance & Efficiency: Latency (p50/p95/p99), throughput (queries/sec), memory footprint.
  - Cost & Dependency: Operational cost estimation per 1k/1M requests, external API dependency tracking.
- **CLI / Programmatic Runners**: Run classification, dataset training/fine-tuning pipelines, and tier comparison reports.
- **Strict Quality Gates**: 100% BDD behavior coverage, >= 95% unit test coverage, 100% mutation test score.

### Out-of-Scope (YAGNI)
- Real-time continuous web scrapers.
- Multi-cloud Kubernetes deployment / distributed cluster autoscalers.
- Web UI / frontend dashboards (results output as JSON / Markdown tables).

## 4. Key Assumptions
- **ASM-GEN-1**: Dataset fits into memory (<100MB RAM); processing runs locally without distributed data systems.
- **ASM-GEN-2**: Canonical taxonomy is 3-class (`POSITIVE`, `NEGATIVE`, `NEUTRAL`) across all model tiers.
- **ASM-GEN-3**: Tier 0 provides offline CPU baseline (zero API cost, <1ms latency).
- **ASM-GEN-4**: Tier 1 evaluations use mock providers in automated CI/CD and require environment API keys for live runs.
- **ASM-GEN-5**: Python 3.10+ execution environment.

## 5. Review Gate
- [x] **Gate 1 Approval**: Manual user sign-off approved.
