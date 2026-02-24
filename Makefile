.PHONY: install lint fmt test eval-smoke report clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package with dev dependencies
	uv sync --extra dev --extra viz

lint: ## Run linting and type checking
	uv run ruff check mmsafe/ tests/ scripts/
	uv run ruff format --check mmsafe/ tests/ scripts/
	uv run mypy mmsafe/

fmt: ## Auto-format code
	uv run ruff check --fix mmsafe/ tests/ scripts/
	uv run ruff format mmsafe/ tests/ scripts/

test: ## Run tests with coverage
	uv run pytest -v --cov=mmsafe --cov-report=term-missing --cov-fail-under=80

test-unit: ## Run unit tests only
	uv run pytest tests/unit/ -v

test-integration: ## Run integration tests only
	uv run pytest tests/integration/ -v

eval-smoke: ## Run evaluation smoke test with stub providers
	uv run mmsafe run --config mmsafe/config/defaults/text_eval.yaml --dry-run

validate-datasets: ## Validate all prompt datasets
	uv run python scripts/validate_datasets.py

report: ## Generate sample HTML report from example results
	uv run mmsafe report artifacts/examples/sample_results.json --format html -o artifacts/examples/report.html

clean: ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
