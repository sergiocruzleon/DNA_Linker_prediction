# DAN_LINKER Makefile
# Quick commands for development and testing

.PHONY: help install test test-unit test-regression benchmark profile clean

# Default target
help:
	@echo "DAN_LINKER - DNA Linker Prediction Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  install        - Install package with dev dependencies"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-regression - Run regression tests only"
	@echo "  benchmark      - Run performance benchmarks"
	@echo "  profile        - Run profiler on probability calculations"
	@echo "  lint           - Run linters (ruff, black, mypy)"
	@echo "  format         - Format code with black"
	@echo "  clean          - Remove cache and temporary files"
	@echo "  run            - Run pipeline on EMD2601 dataset"
	@echo "  run-all        - Run pipeline on all datasets"

# Install
install:
	pip install -e ".[dev]"

# Test targets
test: test-unit test-regression

test-unit:
	pytest tests/test_core_functions.py -v

test-regression:
	pytest tests/test_regression.py -v --timeout=300

# Benchmark targets
benchmark:
	python benchmarks/benchmark_core.py --scaling

profile:
	python benchmarks/benchmark_core.py --profile --particles 50

# Linting
lint:
	ruff check .
	black --check .
	mypy dna_linker/ --ignore-missing-imports || true

format:
	black .
	ruff check --fix .

# Run pipeline
run:
	python scripts/run_pipeline.py --emd 2601 --suffix STA_tmpl --workers 1

run-all:
	python scripts/run_pipeline.py --all --workers 4

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .benchmarks .hypothesis 2>/dev/null || true
	rm -rf benchmark_results 2>/dev/null || true
