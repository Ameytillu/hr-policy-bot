.PHONY: setup run format lint test ingest index

setup:
	python -m pip install -r requirements.txt

run:
	streamlit run streamlit_app.py

ingest:
	python -m src.data_pipeline.cli_ingest --in data/raw_policies --out data/processed

index:
	python scripts/build_index.py

lint:
	python -m ruff check src tests

test:
	pytest -q
