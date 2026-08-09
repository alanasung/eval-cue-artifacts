# Interpreter discovery: never hardcode a single absolute path (X3).
PYTHON_CANDIDATES := \
	python3.12 \
	/usr/local/bin/python3.12 \
	/opt/homebrew/bin/python3.12 \
	/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
	python3

PYTHON := $(shell \
	for c in $(PYTHON_CANDIDATES); do \
		if command -v $$c >/dev/null 2>&1; then \
			$$c -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)' 2>/dev/null && echo $$c && break; \
		elif [ -x "$$c" ]; then \
			$$c -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)' 2>/dev/null && echo $$c && break; \
		fi; \
	done)

ifeq ($(strip $(PYTHON)),)
$(error No Python 3.12 found. Tried: $(PYTHON_CANDIDATES). Install 3.12 or put it on PATH.)
endif

PACKAGE := evalaware
SRC := src
TESTS := tests

.PHONY: install install-dev lint format test test-cov typecheck ci clean pilot smoke doctor help

help:
	@echo "Python: $(PYTHON)"
	@echo "Targets: install install-dev lint format test test-cov typecheck ci clean pilot smoke doctor"

install:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e ".[dev,test]"
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pre_commit install || true

lint:
	$(PYTHON) -m ruff check $(SRC) $(TESTS) scripts

format:
	$(PYTHON) -m ruff format $(SRC) $(TESTS) scripts
	$(PYTHON) -m ruff check --fix $(SRC) $(TESTS) scripts

test:
	$(PYTHON) -m pytest $(TESTS) -v --tb=short -q

test-cov:
	$(PYTHON) -m pytest $(TESTS) -v --tb=short --cov=$(PACKAGE) --cov-report=term-missing --cov-fail-under=60

typecheck:
	$(PYTHON) -m mypy $(SRC)/$(PACKAGE) --ignore-missing-imports

ci: lint test typecheck
	$(PYTHON) -m pytest $(TESTS)/test_sdk.py -v --tb=long
	$(PYTHON) -m pytest $(TESTS) --cov=$(PACKAGE) --cov-fail-under=60 -q

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build *.egg-info
	rm -rf runs results .cache outputs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

pilot:
	$(PYTHON) scripts/run_experiment.py experiment=pilot

smoke:
	$(PYTHON) scripts/run_config_smoke_test.py experiment=smoke

doctor:
	$(PYTHON) scripts/doctor.py
