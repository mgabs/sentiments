.PHONY: help install sync test test-unit test-bdd coverage mutation benchmark baseline clean

help:
	@echo "Available make targets:"
	@echo "  install    Install / sync dependencies via uv"
	@echo "  test       Run full test suite (unit + BDD)"
	@echo "  test-unit  Run unit tests only"
	@echo "  test-bdd   Run BDD behavior scenarios only"
	@echo "  coverage   Run tests with branch coverage (>= 95%)"
	@echo "  mutation   Run mutation testing via mutmut"
	@echo "  baseline   Run live Tier 0 baseline benchmark (alias: benchmark)"
	@echo "  clean      Clean cache and temporary test artifacts"

install sync:
	uv sync

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit

test-bdd:
	uv run pytest tests/bdd

coverage:
	uv run pytest --cov=sentiments --cov-branch

mutation:
	rm -rf mutants/
	uv run mutmut run
	uv run mutmut results

benchmark baseline:
	uv run python scripts/benchmark_baseline.py

clean:
	rm -rf .pytest_cache .coverage htmlcov mutants/ .venv/.mutmut-cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
