# Developer targets: static checks and the CPU test suite.
#
#   make check   ruff check . (falls back to a syntax compile when ruff is missing)
#   make lint    alias of check
#   make test    pytest -m "not gpu" (GPU-marked tests need CUDA + model weights)
#
# PYTHON selects the interpreter used for the fallback compile and for pytest,
# e.g. `make test PYTHON=.venv/bin/python`.

PYTHON ?= python3

.PHONY: check lint test

check:
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check .; \
	elif command -v uvx >/dev/null 2>&1; then \
		uvx ruff check .; \
	else \
		echo "check: ruff not installed - falling back to python syntax compile"; \
		$(PYTHON) -m compileall -q src scripts tests habitat_server evt_bench; \
	fi

lint: check

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q -m "not gpu"
