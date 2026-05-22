# DNA_LINKER Makefile
# Quick commands for local use

.PHONY: help install lint format clean run

# Default target
help:
	@echo "DNA_LINKER - DNA Linker Prediction Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  install        - Install package"
	@echo "  lint           - Run linters (ruff, black, mypy)"
	@echo "  format         - Format code with black"
	@echo "  clean          - Remove cache and temporary files"
	@echo "  run            - Run the example dataset"

# Install
install:
	pip install -e .

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
	python scripts/run_pipeline.py --emd 2601 --motl-file motl_EMD2601_dropped_01.em --workers 1

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
