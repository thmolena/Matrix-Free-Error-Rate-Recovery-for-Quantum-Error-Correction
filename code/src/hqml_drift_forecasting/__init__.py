"""Deterministic structured-operator reproduction package."""
from .saturation import (
    ToggleFit,
    ToggleInterval,
    fit_toggle_rate,
    grouped_toggle_forecast,
    inverse_toggle_probability,
    simultaneous_toggle_interval,
    toggle_probability,
)

__all__ = [
    "ToggleFit",
    "ToggleInterval",
    "fit_toggle_rate",
    "grouped_toggle_forecast",
    "inverse_toggle_probability",
    "reproduce",
    "simultaneous_toggle_interval",
    "toggle_probability",
]


def reproduce() -> dict:
    """Run the locked study without importing the CLI module at package import."""

    from .experiment import reproduce as _reproduce

    return _reproduce()
