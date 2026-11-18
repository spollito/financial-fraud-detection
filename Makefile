PYTHON ?= python

.PHONY: setup lint test
setup:
	$(PYTHON) -m pip install -c constraints.txt -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

test:
	$(PYTHON) -m pytest
