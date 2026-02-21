SHELL := /bin/bash

.PHONY: init run run-demo fmt lint clean

init:
	uv sync --dev

run:
	PYTHONPATH=src uv run python -m semantic_opinion_dynamics.cli run --config config.yaml --network networks/example_org.json --personas personas.yaml

run-demo: run

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

clean:
	rm -rf runs .ruff_cache .pytest_cache
