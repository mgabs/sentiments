# Task Breakdown: Sentiment Analysis Engine & Multi-Tier Benchmarking

**Branch**: `feat/01-sentiment-analysis-core`

## Phase 1: Environment & Project Scaffolding
- [x] 1.1 Initialize project with `uv` (`uv init`, `pyproject.toml`) and install test tools: `pytest`, `pytest-bdd`, `pytest-cov`, `mutmut`.
- [x] 1.2 Establish source directory layout (`src/sentiments/`) and test directories (`tests/bdd/`, `tests/unit/`).
- [x] 1.3 Configure `.gitignore` with `.worktrees/`, `.venv/`, `.coverage`.

## Phase 2: Domain Contracts & Taxonomy (`sentiments.core`)
- [x] 2.1 **BDD**: Write Gherkin feature scenarios for domain models (`SentimentLabel`, `SentimentResult`, `SentimentClassifier` protocol) in `tests/bdd/features/core_contracts.feature`.
- [x] 2.2 **Review Gate 5.1**: Phase 2 BDD scenarios approved.
- [x] 2.3 **Unit Tests**: Implement unit tests for domain models and validation rules (`tests/unit/test_core.py`).
- [x] 2.4 **Implementation**: Implement domain models and protocol definitions in `src/sentiments/core/`.
- [x] 2.5 **Mutation Tests**: Run `mutmut` on `src/sentiments/core/` and ensure 100% kill score.

## Phase 3: Dataset Ingestion & Preprocessing (`sentiments.dataset`) [Parallel Worktree: `.worktrees/dataset-pipeline`]
- [x] 3.1 **BDD**: Write Gherkin feature scenarios for CSV discovery, dynamic delimiter detection, label mapping, and stratified splitting in `tests/bdd/features/dataset_pipeline.feature`.
- [x] 3.2 **Review Gate 5.2**: Phase 3 BDD scenarios approved.
- [x] 3.3 **Unit Tests**: Implement unit tests (`tests/unit/test_dataset.py`) covering edge cases: malformed rows, missing columns, empty files, split ratio validation (>= 95% coverage).
- [x] 3.4 **Implementation**: Implement `DatasetLoader`, `LabelNormalizer`, and `StratifiedSplitter` in `src/sentiments/dataset/`.
- [x] 3.5 **Mutation Tests**: Run `mutmut` on `src/sentiments/dataset/` and achieve 100% kill score.

## Phase 4: Tier 0 Dumb Baseline Classifier (`sentiments.providers.tier0`) [Parallel Worktree: `.worktrees/tier0-baseline`]
- [x] 4.1 **BDD**: Write Gherkin feature scenarios for `MajorityClassClassifier` and `LexiconRuleClassifier` in `tests/bdd/features/tier0_baseline.feature`.
- [x] 4.2 **Review Gate 5.3**: Phase 4 BDD scenarios approved.
- [x] 4.3 **Unit Tests**: Implement unit tests (`tests/unit/test_tier0.py`) verifying latency (<1.0ms), determinism, tie-breaking, and boundary conditions (>= 95% coverage).
- [x] 4.4 **Implementation**: Implement `MajorityClassClassifier` and `LexiconRuleClassifier` in `src/sentiments/providers/tier0.py`.
- [x] 4.5 **Mutation Tests**: Run `mutmut` on `src/sentiments/providers/tier0.py` and achieve 100% kill score.

## Phase 5: Multi-Tier Provider Contracts (`sentiments.providers`)
- [x] 5.1 **BDD**: Write Gherkin feature scenarios for Tier 1–4 provider abstractions, mock adapters, and error handling in `tests/bdd/features/provider_contracts.feature`.
- [x] 5.2 **Review Gate 5.4**: Phase 5 BDD scenarios approved.
- [x] 5.3 **Unit Tests**: Implement unit tests (`tests/unit/test_providers.py`) with mock APIs and timeout handling (>= 95% coverage).
- [x] 5.4 **Implementation**: Implement provider stubs and adapters in `src/sentiments/providers/`.
- [x] 5.5 **Mutation Tests**: Run `mutmut` on `src/sentiments/providers/` and achieve 100% kill score.

## Phase 6: Benchmarking & Reporting Engine (`sentiments.eval`)
- [x] 6.1 **BDD**: Write Gherkin feature scenarios for metric computation (Accuracy, Macro F1, Precision, Recall, Confusion Matrix), latency percentiles, cost estimation, and delta comparison in `tests/bdd/features/benchmarking.feature`.
- [x] 6.2 **Review Gate 5.5**: Present Phase 6 BDD scenarios for approval.
- [x] 6.3 **Unit Tests**: Implement unit tests (`tests/unit/test_eval.py`) covering metric edge cases (zero division, empty arrays) and table rendering (>= 95% coverage).
- [x] 6.4 **Implementation**: Implement `MetricsCalculator`, `CostEstimator`, and `BenchmarkReporter` in `src/sentiments/eval/`.
- [x] 6.5 **Mutation Tests**: Run `mutmut` on `src/sentiments/eval/` and achieve 100% kill score.

## Phase 7: Verification & Final Gate
- [ ] 7.1 Verify 100% BDD scenario pass rate across entire suite.
- [ ] 7.2 Verify >= 95% total unit test line and branch coverage.
- [ ] 7.3 Verify 100% mutation test kill score on all core logic.
- [ ] 7.4 Execute live baseline benchmark against `nlp-datasets/` labeled test split and generate summary report.

---

## Review Gate
- [x] **Gate 4 Approval**: Manual user sign-off approved. Execution started.
