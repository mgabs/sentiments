import time
from pathlib import Path
from sentiments.core import Tier
from sentiments.dataset import DatasetLoader, StratifiedSplitter
from sentiments.providers.tier0 import MajorityClassClassifier, LexiconRuleClassifier
from sentiments.providers.tier2 import Tier2SLMClassifier, OllamaBackend, MLXBackend
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

    # Model 3: Tier 2 Ollama SLM
    ollama_backend = OllamaBackend(model_name="llama3.2:3b")
    if ollama_backend.is_available():
        print(f"[3/4] Running Tier 2 Ollama ({ollama_backend.model_name})...")
        ollama_clf = Tier2SLMClassifier(backend=ollama_backend)
        t0 = time.perf_counter()
        ollama_results = ollama_clf.classify_batch(test_texts)
        ollama_duration = time.perf_counter() - t0
        ollama_preds = [r.label for r in ollama_results]
        ollama_latencies = [r.latency_ms for r in ollama_results]
        ollama_metrics = MetricsCalculator.classification_metrics(y_test_true, ollama_preds)
        ollama_eff = MetricsCalculator.efficiency_metrics(ollama_latencies, total_duration_s=ollama_duration)
        ollama_eval = ModelEvaluation(
            model_name=f"Ollama ({ollama_backend.model_name})",
            tier=Tier.TIER_2,
            classification=ollama_metrics,
            efficiency=ollama_eff,
            estimated_cost_per_1k=0.0,
        )
        evaluations.append(ollama_eval)
    else:
        print("[3/4] Skipping Ollama (daemon not running on http://localhost:11434)")

    # Model 4: Tier 2 MLX SLM
    mlx_backend = MLXBackend(model_name="mlx-community/Llama-3.2-3B-Instruct-4bit")
    if mlx_backend.is_available():
        print(f"[4/4] Running Tier 2 MLX ({mlx_backend.model_name})...")
        mlx_clf = Tier2SLMClassifier(backend=mlx_backend)
        t0 = time.perf_counter()
        mlx_results = mlx_clf.classify_batch(test_texts)
        mlx_duration = time.perf_counter() - t0
        mlx_preds = [r.label for r in mlx_results]
        mlx_latencies = [r.latency_ms for r in mlx_results]
        mlx_metrics = MetricsCalculator.classification_metrics(y_test_true, mlx_preds)
        mlx_eff = MetricsCalculator.efficiency_metrics(mlx_latencies, total_duration_s=mlx_duration)
        mlx_eval = ModelEvaluation(
            model_name=f"MLX ({mlx_backend.model_name.split('/')[-1]})",
            tier=Tier.TIER_2,
            classification=mlx_metrics,
            efficiency=mlx_eff,
            estimated_cost_per_1k=0.0,
        )
        evaluations.append(mlx_eval)
    else:
        print("[4/4] Skipping MLX (requires macOS Apple Silicon and mlx-lm)")

    # Generate Markdown Table
    md_table = BenchmarkReporter.to_markdown(evaluations, baseline=maj_eval)
    footnotes = (
        "### Column Definitions\n"
        "- **Model**: Evaluated classifier name.\n"
        "- **Tier**: Architectural tier (Tier 0: Rule-based heuristics, Tier 2: Local SLM).\n"
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
