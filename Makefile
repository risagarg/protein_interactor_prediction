.PHONY: demo ensemble-demo single-protein-demo test fmt lint clean install

# Demo commands
demo:
	python -m ppi.cli demo

ensemble-demo:
	python -m ppi.cli ensemble-demo

single-protein-demo:
	python -m ppi.cli single-protein-demo

# Development commands
install:
	pip install -e .

install-dev:
	pip install -e .[dev]

test:
	pytest -v

test-cov:
	pytest --cov=src/ppi --cov-report=html

fmt:
	black src tests
	ruff check --fix src tests

lint:
	ruff check src tests
	mypy src

# Clean up
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf artifacts/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Help
help:
	@echo "Available commands:"
	@echo "  demo              - Run basic PPI prediction demo"
	@echo "  ensemble-demo     - Run ensemble model demo"
	@echo "  single-protein-demo - Run single protein prediction demo"
	@echo "  test              - Run tests"
	@echo "  test-cov          - Run tests with coverage"
	@echo "  fmt               - Format code with black and ruff"
	@echo "  lint              - Lint code with ruff and mypy"
	@echo "  clean             - Clean up build artifacts"
	@echo "  install           - Install package in development mode"
	@echo "  install-dev       - Install with development dependencies"
