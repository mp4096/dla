.DEFAULT_GOAL := help

.PHONY: install-dev
install-dev: ## Dev install
	python setup.py develop

.PHONY: init
init: ## Install dependencies
	pip install -r requirements.txt

.PHONY: init-dev
init-dev: ## Install dev dependencies
	pip install -r requirements-dev.txt

.PHONY: lint
lint: ## Lint code for PEP8 and PEP257 conformity
	pycodestyle ./dla/
	pydocstyle --convention=numpy ./dla/

.PHONY: test
test: ## Run unit tests
	pytest -v

.PHONY: test-coverage
test-coverage: ## Run unit tests with coverage
	pytest --cov-config .coveragerc --cov=dla -v

.PHONY: test-coverage-html
test-coverage-html: ## Run unit tests with coverage, create a HTML report
	pytest --cov-config .coveragerc --cov=dla --cov-report=html -v

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'
