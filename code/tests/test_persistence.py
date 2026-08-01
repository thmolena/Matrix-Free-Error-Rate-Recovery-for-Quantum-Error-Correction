from __future__ import annotations

import numpy as np

from hqml_drift_forecasting.persistence import (
    drift_error_bound,
    grouped_effective_toggle_persistence,
    grouped_last_rate,
)
from hqml_drift_forecasting.saturation import toggle_probability


def test_persistence_is_exact_for_stationary_toggle_groups() -> None:
    rounds = np.array([1, 3, 5, 1, 3, 5])
    groups = ["a", "a", "a", "b", "b", "b"]
    rates = np.concatenate(
        (toggle_probability(0.03, rounds[:3]), toggle_probability(0.08, rounds[3:]))
    )
    known = rounds <= 3
    prediction, fits = grouped_effective_toggle_persistence(rounds, rates, groups, known)
    np.testing.assert_allclose(prediction, rates, atol=1e-14)
    assert fits["a"].last_round == 3
    assert fits["b"].last_round == 3


def test_persistence_bound_covers_declared_toggle_drift() -> None:
    q0 = 0.04
    future_rounds = np.array([7, 9, 11])
    drift = np.array([0.001, -0.0015, 0.002])
    prediction = toggle_probability(q0, future_rounds)
    observed = np.array(
        [toggle_probability(q0 + delta, [r])[0] for delta, r in zip(drift, future_rounds)]
    )
    assert np.all(
        np.abs(observed - prediction) <= drift_error_bound(future_rounds, 0.002)
    )


def test_last_rate_baseline_uses_latest_round_only() -> None:
    rounds = np.array([1, 3, 5])
    rates = np.array([0.1, 0.2, 0.3])
    prediction = grouped_last_rate(rounds, rates, [0, 0, 0], rounds <= 3)
    np.testing.assert_allclose(prediction, 0.2)
