.PHONY: setup run format lint test ingest index

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run:
	streamlit run src/app/main.py

ingest:
	python -m src.data_pipeline.cli_ingest --in data/raw_policies --out data/processed

index:
	python scripts/build_index.py

lint:
	python -m pyflakes src || true

test:
	pytest -q
