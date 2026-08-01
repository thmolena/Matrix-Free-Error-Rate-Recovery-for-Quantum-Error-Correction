from __future__ import annotations

import numpy as np

from hqml_drift_forecasting import experiment
from hqml_drift_forecasting.saturation import (
    fit_toggle_rate,
    grouped_toggle_forecast,
    simultaneous_toggle_interval,
    toggle_probability,
)


def test_toggle_semigroup_and_depth_sensitivity() -> None:
    for q in (0.0, 0.007, 0.031, 0.2, 0.5):
        for r, s in ((1, 2), (3, 8), (11, 14)):
            p_r = float(toggle_probability(q, [r])[0])
            p_s = float(toggle_probability(q, [s])[0])
            p_sum = float(toggle_probability(q, [r + s])[0])
            np.testing.assert_allclose(
                p_sum, p_r + p_s - 2.0 * p_r * p_s, atol=2e-15
            )
    for depth in (1, 7, 25):
        for q_left, q_right in ((0.0, 0.01), (0.03, 0.031), (0.2, 0.49)):
            difference = abs(
                float(toggle_probability(q_right, [depth])[0])
                - float(toggle_probability(q_left, [depth])[0])
            )
            assert difference <= depth * abs(q_right - q_left) + 2e-15


def test_grid_fit_recovers_exact_toggle_curve_and_reports_gap() -> None:
    rounds = np.asarray([1, 3, 5, 9, 13], dtype=int)
    truth_q = 0.0375
    rates = toggle_probability(truth_q, rounds)
    fit = fit_toggle_rate(rounds, rates, grid_points=20_001)
    assert fit.q == truth_q
    assert fit.mean_squared_error < 1e-30
    assert fit.grid_objective_gap_bound == 13.0 / 80_000.0


def test_simultaneous_interval_covers_compatible_curve() -> None:
    rounds = np.asarray([1, 3, 5, 9, 13], dtype=int)
    truth_q = 0.026
    rates = toggle_probability(truth_q, rounds)
    interval = simultaneous_toggle_interval(
        rounds, rates, np.full(len(rounds), 50_000), delta=0.05
    )
    assert interval.compatible
    assert interval.q_lower <= truth_q <= interval.q_upper


def test_physical_saturation_baseline_falsifies_old_headline() -> None:
    raw, target, rounds, rows = experiment._hardware_arrays()
    del raw
    groups = [
        (row["basis"], row["distance"], row["center_row"], row["center_col"])
        for row in rows
    ]
    folds = (
        (11, 13, 15, 2.092e-4),
        (13, 15, 17, 1.090e-3),
        (15, 17, 19, 4.631e-4),
        (17, 19, 25, 5.872e-4),
    )
    observed = []
    for development_round, test_low, test_high, old_method_mse in folds:
        prediction, _ = grouped_toggle_forecast(
            rounds,
            target,
            groups,
            rounds <= development_round,
        )
        test = (rounds >= test_low) & (rounds <= test_high)
        mse = float(np.mean((prediction[test] - target[test]) ** 2))
        observed.append(mse)
        assert mse < old_method_mse
    np.testing.assert_allclose(
        observed,
        [
            1.14922173041939e-4,
            1.35487545960105e-4,
            1.39341638641522e-4,
            1.40319291535874e-4,
        ],
        rtol=2e-12,
        atol=2e-15,
    )


def test_hardware_rows_reject_stationary_toggle_hypothesis() -> None:
    _, target, rounds, rows = experiment._hardware_arrays()
    groups = [
        (row["basis"], row["distance"], row["center_row"], row["center_col"])
        for row in rows
    ]
    shots = np.asarray([int(row["shots"]) for row in rows], dtype=int)
    for group in dict.fromkeys(groups):
        in_group = np.asarray([item == group for item in groups], dtype=bool)
        training = in_group & (rounds <= 17)
        interval = simultaneous_toggle_interval(
            rounds[training], target[training], shots[training], delta=0.005
        )
        assert not interval.compatible
