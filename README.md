# Effective-Toggle Persistence for Rolling Forecasts of Surface-Code Logical Error Rates

This repository contains the manuscript, authenticated hardware record, and CPU implementation for a rolling logical-error forecast based on effective-toggle persistence.

The estimator takes the latest measured logical error rate within each hardware stratum, inverts the odd-parity accumulation law, and propagates the resulting effective per-round toggle to later circuit depths. It requires no numerical fitting.

## Registered evidence

- 130 Google surface-code hardware experiments.
- 50,000 shots per experiment.
- Two logical bases, two code distances, thirteen round settings, and ten hardware strata.
- Tensor-network decoder rates used for development.
- Belief-matching decoder rates used as a separate confirmation endpoint.
- Four rolling folds with training restricted to earlier circuit depths.

Effective-toggle persistence reduced mean-squared error against the strongest registered comparator in every fold: 35.8–61.2 percent on development and 13.0–58.4 percent on confirmation.

## Reproduction

```bash
python -m venv /tmp/effective-toggle-reproduction
/tmp/effective-toggle-reproduction/bin/pip install -e "code[test]"
/tmp/effective-toggle-reproduction/bin/effective-toggle-study \
  --output /tmp/effective-toggle-study
/tmp/effective-toggle-reproduction/bin/python -m pytest -q code/tests
tectonic main.tex
```

## Evidence map

- `main.tex` and `main.pdf`: paper and compiled manuscript.
- `code/data/source/google_qec3v5_experiment_summary.csv`: authenticated hardware summary.
- `code/src/hqml_drift_forecasting/persistence.py`: estimator and drift envelope.
- `code/src/hqml_drift_forecasting/persistence_study.py`: rolling evaluation and figure generation.
- `code/results/persistence/persistence_study.json`: complete scientific and timing record.
- `code/results/persistence/rolling_metrics.csv`: fold-level metrics.
- `code/results/persistence/figures/`: generated paper figures.
- `code/tests/test_persistence.py`: exactness, drift, and baseline tests.

The portable semantic digest is `fb0ff77517c4acb9f59104dbc10cb125e5b992f4e45076afdc7cf7aae070cc6c`.

## Scope

The result concerns aggregate logical error-rate forecasting on one public hardware campaign. It does not decode individual syndromes and does not establish transfer across devices, calibration dates, or code families.
