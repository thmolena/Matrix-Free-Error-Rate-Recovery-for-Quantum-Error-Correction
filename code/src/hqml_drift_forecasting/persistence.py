"""Effective-toggle persistence for rolling logical-error forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np

from .saturation import inverse_toggle_probability, toggle_probability


@dataclass(frozen=True)
class PersistenceFit:
    group: Hashable
    last_round: int
    last_rate: float
    effective_toggle: float
    training_points: int


def grouped_effective_toggle_persistence(
    rounds: np.ndarray | Sequence[int],
    rates: np.ndarray | Sequence[float],
    groups: Sequence[Hashable],
    known_mask: np.ndarray | Sequence[bool],
) -> tuple[np.ndarray, dict[Hashable, PersistenceFit]]:
    """Forecast each group from its latest observed effective toggle rate."""

    depth = np.asarray(rounds, dtype=int)
    target = np.asarray(rates, dtype=float)
    known = np.asarray(known_mask, dtype=bool)
    if depth.ndim != 1 or target.shape != depth.shape or known.shape != depth.shape:
        raise ValueError("rounds, rates, and known_mask must have identical shapes")
    if len(groups) != len(depth):
        raise ValueError("groups has the wrong length")
    if np.any(depth < 1) or np.any(~np.isfinite(target)):
        raise ValueError("depths must be positive and rates must be finite")
    if np.any((target < 0.0) | (target > 0.5)):
        raise ValueError("rates must lie in [0, 1/2]")

    prediction = np.empty(len(depth), dtype=float)
    fits: dict[Hashable, PersistenceFit] = {}
    for group in dict.fromkeys(groups):
        in_group = np.fromiter(
            (item == group for item in groups), dtype=bool, count=len(depth)
        )
        training = in_group & known
        if not np.any(training):
            raise ValueError(f"group {group!r} has no known rows")
        last_round = int(np.max(depth[training]))
        latest = training & (depth == last_round)
        last_rate = float(np.mean(target[latest]))
        effective = inverse_toggle_probability(last_rate, last_round)
        prediction[in_group] = toggle_probability(effective, depth[in_group])
        fits[group] = PersistenceFit(
            group=group,
            last_round=last_round,
            last_rate=last_rate,
            effective_toggle=effective,
            training_points=int(np.sum(training)),
        )
    return prediction, fits


def grouped_last_rate(
    rounds: np.ndarray | Sequence[int],
    rates: np.ndarray | Sequence[float],
    groups: Sequence[Hashable],
    known_mask: np.ndarray | Sequence[bool],
) -> np.ndarray:
    """Strong nearest-round baseline with no saturation transformation."""

    depth = np.asarray(rounds, dtype=int)
    target = np.asarray(rates, dtype=float)
    known = np.asarray(known_mask, dtype=bool)
    prediction = np.empty(len(depth), dtype=float)
    for group in dict.fromkeys(groups):
        in_group = np.fromiter(
            (item == group for item in groups), dtype=bool, count=len(depth)
        )
        training = in_group & known
        if not np.any(training):
            raise ValueError(f"group {group!r} has no known rows")
        last_round = int(np.max(depth[training]))
        prediction[in_group] = float(np.mean(target[training & (depth == last_round)]))
    return prediction


def drift_error_bound(rounds: np.ndarray | Sequence[int], drift_radius: float) -> np.ndarray:
    """Return ``r * drift_radius`` from the toggle-map Lipschitz theorem."""

    depth = np.asarray(rounds, dtype=int)
    radius = float(drift_radius)
    if np.any(depth < 1) or not np.isfinite(radius) or radius < 0.0:
        raise ValueError("positive depths and a finite nonnegative radius are required")
    return depth.astype(float) * radius
