import time
from pathlib import Path
from sentiments.core import Tier
from sentiments.dataset import DatasetLoader, StratifiedSplitter
from sentiments.providers.tier0 import MajorityClassClassifier, LexiconRuleClassifier
from sentiments.providers.tier2 import Tier2SLMClassifier, OllamaBackend, MLXBackend
from sentiments.providers.tier4 import ONNXClassifier
from sentiments.eval import MetricsCalculator, BenchmarkReporter, ModelEvaluation


def run_benchmark():
    datasets_dir = Path("nlp-datasets/trustpilot-reviews")
    print(f"Loading datasets from {datasets_dir}...")
    records = DatasetLoader(datasets_dir).load_all()
    print(f"Loaded {len(records)} valid records.")

    splitter = StratifiedSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    split = splitter.split(records)
    train_data, val_data, test_data = split.train, split.val, split.test
    print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    y_test_true = [r.label for r in test_data]
    test_texts = [r.text for r in test_data]

    # Model 1: Majority Class Classifier (Tier 0 baseline)
    print("\n[1/4] Running MajorityClassClassifier...")
    majority_clf = MajorityClassClassifier()
    majority_clf.fit(train_data)

    t0 = time.perf_counter()
    majority_results = majority_clf.classify_batch(test_texts)
    maj_duration = time.perf_counter() - t0
    maj_preds = [r.label for r in majority_results]
    maj_latencies = [r.latency_ms for r in majority_results]

    maj_metrics = MetricsCalculator.classification_metrics(y_test_true, maj_preds)
    maj_eff = MetricsCalculator.efficiency_metrics(maj_latencies, total_duration_s=maj_duration)
    maj_eval = ModelEvaluation(
        model_name="MajorityClassClassifier",
        tier=Tier.TIER_0,
        classification=maj_metrics,
        efficiency=maj_eff,
        estimated_cost_per_1k=0.0,
    )

    # Model 2: Lexicon Rule Classifier (Tier 0)
    print("[2/4] Running LexiconRuleClassifier...")
    lexicon_clf = LexiconRuleClassifier()

    t0 = time.perf_counter()
    lex_results = lexicon_clf.classify_batch(test_texts)
    lex_duration = time.perf_counter() - t0
    lex_preds = [r.label for r in lex_results]
    lex_latencies = [r.latency_ms for r in lex_results]

    lex_metrics = MetricsCalculator.classification_metrics(y_test_true, lex_preds)
    lex_eff = MetricsCalculator.efficiency_metrics(lex_latencies, total_duration_s=lex_duration)
    lex_eval = ModelEvaluation(
        model_name="LexiconRuleClassifier",
        tier=Tier.TIER_0,
        classification=lex_metrics,
        efficiency=lex_eff,
        estimated_cost_per_1k=0.0,
    )

    evaluations = [maj_eval, lex_eval]

    # Helper function to evaluate and report SLM models
    def run_slm_eval(name: str, clf: Tier2SLMClassifier) -> None:
        print(f"Running {name}...")
        t0 = time.perf_counter()
        results = clf.classify_batch(test_texts)
        duration = time.perf_counter() - t0
        preds = [r.label for r in results]
        latencies = [r.latency_ms for r in results]
        metrics = MetricsCalculator.classification_metrics(y_test_true, preds)
        eff = MetricsCalculator.efficiency_metrics(latencies, total_duration_s=duration)
        evaluations.append(
            ModelEvaluation(
                model_name=name,
                tier=Tier.TIER_2,
                classification=metrics,
                efficiency=eff,
                estimated_cost_per_1k=0.0,
            )
        )

    # Tier 2 Ollama models
    for model in ["llama3.2:3b", "qwen2.5:3b"]:
        ollama_backend = OllamaBackend(model_name=model)
        if ollama_backend.is_available():
            run_slm_eval(f"Ollama ({model})", Tier2SLMClassifier(backend=ollama_backend))
        else:
            print(f"Skipping Ollama ({model}): daemon unreachable")

    # Tier 2 MLX models
    for model_id in ["mlx-community/Llama-3.2-3B-Instruct-4bit", "mlx-community/Qwen2.5-3B-Instruct-4bit"]:
        mlx_backend = MLXBackend(model_name=model_id)
        if mlx_backend.is_available():
            short_name = model_id.split("/")[-1]
            run_slm_eval(f"MLX ({short_name})", Tier2SLMClassifier(backend=mlx_backend))
        else:
            print(f"Skipping MLX ({model_id}): platform/library unavailable")

    # Tier 4 Quantized ONNX Model
    tier4_model_dir = Path("models/tier4")
    if tier4_model_dir.exists():
        print(f"\n[Tier 4] Running ONNXClassifier ({tier4_model_dir})...")
        onnx_clf = ONNXClassifier(model_path=tier4_model_dir)
        t0 = time.perf_counter()
        onnx_results = onnx_clf.classify_batch(test_texts)
        onnx_duration = time.perf_counter() - t0
        onnx_preds = [r.label for r in onnx_results]
        onnx_latencies = [r.latency_ms for r in onnx_results]
        onnx_metrics = MetricsCalculator.classification_metrics(y_test_true, onnx_preds)
        onnx_eff = MetricsCalculator.efficiency_metrics(onnx_latencies, total_duration_s=onnx_duration)
        evaluations.append(
            ModelEvaluation(
                model_name=f"ONNX ({onnx_clf.model_name})",
                tier=Tier.TIER_4,
                classification=onnx_metrics,
                efficiency=onnx_eff,
                estimated_cost_per_1k=0.0,
            )
        )
    else:
        print(f"Skipping Tier 4: {tier4_model_dir} not found")

    # Generate Markdown Table
    md_table = BenchmarkReporter.to_markdown(evaluations, baseline=maj_eval)
    footnotes = (
        "### Column Definitions\n"
        "- **Model**: Evaluated classifier name.\n"
        "- **Tier**: Architectural tier (Tier 0: Rule-based heuristics, Tier 2: Local SLM, Tier 4: Quantized ONNX CPU).\n"
        "- **Accuracy**: Proportion of correct predictions over total samples.\n"
        "- **Macro F1**: Unweighted mean of F1-scores across all classes.\n"
        "- **Δ F1**: Difference in Macro F1 relative to baseline (`Macro F1(Candidate) - Macro F1(Baseline)`).\n"
        "- **p50 Latency (ms)**: Median inference latency per sample in milliseconds.\n"
        "- **Latency Ratio**: Relative latency multiplier compared to baseline (`p50(Candidate) / p50(Baseline)`).\n"
        "- **QPS**: Throughput measured in queries processed per second.\n"
        "- **Cost / 1k ($)**: Estimated operational cost in USD per 1,000 predictions.\n"
        "- **Δ Cost ($)**: Cost delta per 1,000 predictions relative to baseline (`Cost(Candidate) - Cost(Baseline)`)."
    )
    print("\n--- BENCHMARK REPORT ---")
    print(md_table)
    print(f"\n{footnotes}")

    output_path = Path("benchmark_report.md")
    output_path.write_text(
        f"# Multi-Tier Sentiment Benchmark Report\n\n"
        f"Dataset: `nlp-datasets/trustpilot-reviews` ({len(test_data)} test samples)\n\n"
        f"{md_table}\n\n"
        f"{footnotes}\n"
    )
    print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    run_benchmark()
