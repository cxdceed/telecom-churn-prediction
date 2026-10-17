.PHONY: all data clean features eda train evaluate sql notebooks test lint app clean-outputs
PY ?= python

all: data clean features eda train evaluate sql notebooks   ## reproduce everything

data:      ; $(PY) -m src.generate_data
clean:     ; $(PY) -m src.clean
features:  ; $(PY) -m src.features
eda:       ; $(PY) -m src.eda
train:     ; $(PY) -m src.train
evaluate:  ; $(PY) -m src.evaluate
sql:       ; $(PY) -m src.build_sqlite
notebooks: ; $(PY) -m src.run_notebooks
test:      ; $(PY) -m pytest -q
lint:      ; ruff check src tests app
app:       ; streamlit run app/streamlit_app.py
clean-outputs: ; rm -rf data/raw/*.csv data/processed/* models/*.joblib reports/*.json reports/*.csv reports/figures/*.png
