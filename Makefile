PY := python

.PHONY: all download data features models evaluate bundle app test clean

all: data features models evaluate bundle

download:
	$(PY) scripts/download.py

data:
	$(PY) -m ballpark.ingest

features:
	$(PY) -m ballpark.state

models:
	$(PY) -m ballpark.models.outcome
	$(PY) -m ballpark.models.winprob
	$(PY) -m ballpark.models.impact
	$(PY) -m ballpark.models.matchup

evaluate:
	$(PY) -m ballpark.evaluate

bundle:
	$(PY) scripts/build_app_bundle.py

app:
	streamlit run app/Home.py

test:
	$(PY) -m pytest tests -q

clean:
	rm -rf data/interim/* data/processed/* models/*.joblib models/*_metrics.json
