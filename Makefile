.PHONY: help install test lint typecheck build

help:
\t@echo "Targets:"
\t@echo "  make install     - install package in editable mode with dev deps"
\t@echo "  make test        - run pytest"
\t@echo "  make lint        - run flake8"
\t@echo "  make typecheck   - run mypy"
\t@echo "  make build       - build wheel"

install:
\tpython -m pip install --upgrade pip
\tpython -m pip install -e .[dev]

test:
\tpytest -q

lint:
\tflake8 src tests

typecheck:
\tmypy src

build:
\tpython -m build
