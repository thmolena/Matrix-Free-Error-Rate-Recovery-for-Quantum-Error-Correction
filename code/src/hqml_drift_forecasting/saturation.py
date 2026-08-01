"""Auditable saturation baselines for repeated logical-flip experiments.

If an observable flips with probability ``q`` on each independent round, its
odd-parity probability after ``r`` rounds is ``(1 - (1 - 2 q)**r) / 2``.
These routines fit that model on a declared grid, expose the grid-optimization
certificate, and construct a simultaneous finite-shot compatibility interval.
They never inspect a held-out target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class ToggleFit:
    """One group's fitted toggle rate and deterministic grid certificate."""

    q: float
    mean_squared_error: float
    grid_points: int
    grid_objective_gap_bound: float
    training_points: int


@dataclass(frozen=True)
class ToggleInterval:
    """Simultaneous Hoeffding interval after monotone inversion."""

    q_lower: float
    q_upper: float
    confidence: float
    compatible: bool


def toggle_probability(q: float, rounds: np.ndarray | Sequence[int]) -> np.ndarray:
    """Return the odd-parity probability after independent binary toggles."""

    q_value = float(q)
    if not np.isfinite(q_value) or not 0.0 <= q_value <= 0.5:
        raise ValueError("q must be finite and lie in [0, 1/2]")
    depth = np.asarray(rounds, dtype=int)
    if np.any(depth < 1):
        raise ValueError("round counts must be positive integers")
    return 0.5 * (1.0 - np.power(1.0 - 2.0 * q_value, depth))


def inverse_toggle_probability(probability: float, rounds: int) -> float:
    """Invert the toggle curve on ``q in [0, 1/2]``."""

    p = float(probability)
    depth = int(rounds)
    if not np.isfinite(p) or not 0.0 <= p <= 0.5:
        raise ValueError("probability must be finite and lie in [0, 1/2]")
    if depth < 1:
        raise ValueError("rounds must be positive")
    return 0.5 * (1.0 - (1.0 - 2.0 * p) ** (1.0 / depth))


def fit_toggle_rate(
    rounds: np.ndarray | Sequence[int],
    rates: np.ndarray | Sequence[float],
    *,
    weights: np.ndarray | Sequence[float] | None = None,
    grid_points: int = 20_001,
) -> ToggleFit:
    """Fit a stationary toggle rate on a uniform endpoint-inclusive grid.

    For targets in ``[0, 1/2]``, the weighted mean-square objective is
    ``r_max``-Lipschitz in ``q``.  A nearest point on the grid is therefore at
    most ``r_max/(4(K-1))`` above the continuous optimum.
    """

    depth = np.asarray(rounds, dtype=int)
    target = np.asarray(rates, dtype=float)
    if depth.ndim != 1 or target.shape != depth.shape or len(depth) == 0:
        raise ValueError("rounds and rates must be nonempty one-dimensional arrays")
    if np.any(depth < 1):
        raise ValueError("round counts must be positive")
    if np.any(~np.isfinite(target)) or np.any((target < 0.0) | (target > 0.5)):
        raise ValueError("rates must be finite and lie in [0, 1/2]")
    if grid_points < 2:
        raise ValueError("grid_points must be at least two")
    if weights is None:
        normalized = np.full(len(depth), 1.0 / len(depth))
    else:
        raw_weights = np.asarray(weights, dtype=float)
        if raw_weights.shape != depth.shape:
            raise ValueError("weights have the wrong shape")
        if np.any(~np.isfinite(raw_weights)) or np.any(raw_weights < 0.0):
            raise ValueError("weights must be finite and nonnegative")
        total = float(raw_weights.sum())
        if total <= 0.0:
            raise ValueError("weights must have positive sum")
        normalized = raw_weights / total

    best_q = 0.0
    best_loss = float("inf")
    # Stream the grid so fitting uses O(number of observations) memory.
    for q_value in np.linspace(0.0, 0.5, int(grid_points)):
        residual = toggle_probability(float(q_value), depth) - target
        loss = float(np.dot(normalized, residual * residual))
        if loss < best_loss:
            best_loss = loss
            best_q = float(q_value)
    gap = float(np.max(depth)) / (4.0 * (int(grid_points) - 1))
    return ToggleFit(
        q=best_q,
        mean_squared_error=best_loss,
        grid_points=int(grid_points),
        grid_objective_gap_bound=gap,
        training_points=len(depth),
    )


def grouped_toggle_forecast(
    rounds: np.ndarray | Sequence[int],
    rates: np.ndarray | Sequence[float],
    groups: Sequence[Hashable],
    known_mask: np.ndarray | Sequence[bool],
    *,
    weights: np.ndarray | Sequence[float] | None = None,
    grid_points: int = 20_001,
) -> tuple[np.ndarray, dict[Hashable, ToggleFit]]:
    """Fit each declared stratum using known rows and predict every row."""

    depth = np.asarray(rounds, dtype=int)
    target = np.asarray(rates, dtype=float)
    known = np.asarray(known_mask, dtype=bool)
    if target.shape != depth.shape or known.shape != depth.shape:
        raise ValueError("rounds, rates, and known_mask must have identical shapes")
    if len(groups) != len(depth):
        raise ValueError("groups has the wrong length")
    raw_weights = None if weights is None else np.asarray(weights, dtype=float)
    if raw_weights is not None and raw_weights.shape != depth.shape:
        raise ValueError("weights have the wrong shape")

    ordered_groups = list(dict.fromkeys(groups))
    prediction = np.empty(len(depth), dtype=float)
    fits: dict[Hashable, ToggleFit] = {}
    for group in ordered_groups:
        in_group = np.fromiter(
            (item == group for item in groups), dtype=bool, count=len(depth)
        )
        training = in_group & known
        if not np.any(training):
            raise ValueError(f"group {group!r} has no known rows")
        fit = fit_toggle_rate(
            depth[training],
            target[training],
            weights=None if raw_weights is None else raw_weights[training],
            grid_points=grid_points,
        )
        fits[group] = fit
        prediction[in_group] = toggle_probability(fit.q, depth[in_group])
    return prediction, fits


def simultaneous_toggle_interval(
    rounds: np.ndarray | Sequence[int],
    observed_rates: np.ndarray | Sequence[float],
    shots: np.ndarray | Sequence[int],
    *,
    delta: float = 0.05,
) -> ToggleInterval:
    """Return a simultaneous finite-shot interval for a stationary ``q``.

    A union bound and Hoeffding's inequality give simultaneous rate intervals
    with probability at least ``1-delta``.  Monotone inversion maps each rate
    interval to an interval for ``q``.  An empty intersection rejects the
    stationary independent-toggle hypothesis at the declared level; it is not
    silently widened.
    """

    depth = np.asarray(rounds, dtype=int)
    rate = np.asarray(observed_rates, dtype=float)
    counts = np.asarray(shots, dtype=int)
    if depth.ndim != 1 or rate.shape != depth.shape or counts.shape != depth.shape:
        raise ValueError("rounds, observed_rates, and shots must have identical shapes")
    if len(depth) == 0 or np.any(depth < 1) or np.any(counts < 1):
        raise ValueError("at least one positive depth and shot count is required")
    if np.any(~np.isfinite(rate)) or np.any((rate < 0.0) | (rate > 0.5)):
        raise ValueError("observed rates must be finite and lie in [0, 1/2]")
    if not np.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")

    epsilon = np.sqrt(np.log(2.0 * len(depth) / delta) / (2.0 * counts))
    lower_rates = np.maximum(0.0, rate - epsilon)
    upper_rates = np.minimum(0.5, rate + epsilon)
    lowers = np.asarray(
        [inverse_toggle_probability(p, int(r)) for p, r in zip(lower_rates, depth)]
    )
    uppers = np.asarray(
        [inverse_toggle_probability(p, int(r)) for p, r in zip(upper_rates, depth)]
    )
    q_lower = float(np.max(lowers))
    q_upper = float(np.min(uppers))
    return ToggleInterval(
        q_lower=q_lower,
        q_upper=q_upper,
        confidence=1.0 - float(delta),
        compatible=bool(q_lower <= q_upper),
    )

