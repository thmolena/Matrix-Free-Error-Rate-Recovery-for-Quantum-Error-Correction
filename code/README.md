# hqml-drift-forecasting

This wheel reproduces the saturation-baseline audit in the paper **A
Saturation-Baseline Audit of Matrix-Free Quantum Error-Rate Recovery**.

The primary hardware experiment compares three frozen predictors on four
rolling-origin splits:

1. linear ridge extrapolation;
2. the legacy depth-monotone rank-counterfactual matrix-free blend; and
3. a groupwise stationary binary-toggle saturation curve.

The third predictor beats the legacy blend on every split. Replacing ridge by
the toggle baseline in the same matrix-free selector yields `alpha = 0` in all
four folds. The package preserves that contradiction as a regression test.

`saturation.py` implements

```text
p_r(q) = (1 - (1 - 2q)^r) / 2,
```

a streamed uniform-grid fit with a deterministic continuous-objective gap
bound, and a simultaneous Hoeffding interval after monotone inversion. All ten
hardware strata reject a common stationary rate at familywise level 0.05, so
the curve is reported as a strong predictive baseline, not a validated noise
model.

## Install and verify

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
drift-forecasting-reproduce --verify
python -m pytest -q tests
```

Regenerate six manuscript figures and the three numeric tables used by the paper:

```bash
drift-forecasting-figures \
  --figure-dir manuscript_assets/figures \
  --table-dir manuscript_assets/tables
```

## Key files

- `src/hqml_drift_forecasting/saturation.py`: toggle law, fit, and interval.
- `src/hqml_drift_forecasting/experiment.py`: frozen folds, legacy comparator,
  audited selector, simulated stress study, and locked output.
- `src/hqml_drift_forecasting/matrix_free.py`: entry oracle and hierarchy.
- `src/hqml_drift_forecasting/figures.py`: generated line/bar figures and
  numeric tables.
- `tests/test_saturation.py`: theorem and falsification regressions.
- `tests/test_reproduction.py`: provenance, replay, and matrix-free gates.

The Google hardware-derived table and the simulated Zenodo archive remain
explicitly separated. Counted products and stored scalars are not elapsed-time
measurements.

<!-- standardized-public-entry-points -->
## Standard public entry points

From the repository root:

```bash
python -m pip install -e code
python code/scripts/download_data.py
python code/scripts/reproduce.py
python code/scripts/make_figures.py
python code/scripts/validate_release.py
```
