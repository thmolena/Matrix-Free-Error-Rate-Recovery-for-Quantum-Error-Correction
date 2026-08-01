"""Rolling hardware study for effective-toggle persistence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .persistence import grouped_effective_toggle_persistence, grouped_last_rate
from .saturation import grouped_toggle_forecast, inverse_toggle_probability

FOLDS = ((9, 11, 13, 15), (11, 13, 15, 17), (13, 15, 17, 19), (15, 17, 19, 25))
ENDPOINTS = (
    ("tensor_network_error_rate", "development decoder"),
    ("belief_matching_error_rate", "confirmation decoder"),
)
SOURCE_SHA256 = "15ffb8c7773a4fa5d1d498bbe01fd929334c0c13da69ab81704ad13beadde5f5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path):
    if _sha256(path) != SOURCE_SHA256:
        raise ValueError("hardware summary SHA-256 mismatch")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rounds = np.asarray([int(row["rounds"]) for row in rows], dtype=int)
    groups = [
        (row["basis"], int(row["distance"]), int(row["center_row"]), int(row["center_col"]))
        for row in rows
    ]
    features = np.asarray(
        [
            [
                1.0,
                float(row["basis"] == "Z"),
                float(int(row["distance"]) == 5),
                int(row["rounds"]) / 25.0,
                int(row["center_row"]) / 7.0,
                int(row["center_col"]) / 7.0,
            ]
            for row in rows
        ],
        dtype=float,
    )
    return rows, rounds, groups, features


def _ridge_prediction(features: np.ndarray, target: np.ndarray, train: np.ndarray) -> np.ndarray:
    penalty = 0.01 * np.eye(features.shape[1])
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(
        features[train].T @ features[train] + penalty,
        features[train].T @ target[train],
    )
    return np.clip(features @ coefficient, 0.0, 0.5)


def _metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    residual = prediction[mask] - target[mask]
    return {
        "mse": float(np.mean(residual * residual)),
        "mae": float(np.mean(np.abs(residual))),
        "maximum_absolute_error": float(np.max(np.abs(residual))),
    }


def _endpoint_study(
    endpoint: str,
    role: str,
    rows: list[dict[str, str]],
    rounds: np.ndarray,
    groups,
    features: np.ndarray,
) -> dict[str, Any]:
    target = np.asarray([float(row[endpoint]) for row in rows], dtype=float)
    fold_records: list[dict[str, Any]] = []
    for fit_max, development_round, test_low, test_high in FOLDS:
        train = rounds <= development_round
        test = (rounds >= test_low) & (rounds <= test_high)
        persistence, fits = grouped_effective_toggle_persistence(
            rounds, target, groups, train
        )
        last_rate = grouped_last_rate(rounds, target, groups, train)
        stationary, stationary_fits = grouped_toggle_forecast(
            rounds, target, groups, train
        )
        ridge = _ridge_prediction(features, target, train)
        predictions = {
            "effective_toggle_persistence": persistence,
            "last_observed_rate": last_rate,
            "stationary_toggle_fit": stationary,
            "ridge_extrapolation": ridge,
        }
        methods = {name: _metrics(value, target, test) for name, value in predictions.items()}
        strongest_baseline = min(
            (name for name in methods if name != "effective_toggle_persistence"),
            key=lambda name: methods[name]["mse"],
        )
        proposal_error = (persistence[test] - target[test]) ** 2
        baseline_error = (predictions[strongest_baseline][test] - target[test]) ** 2
        fold_records.append(
            {
                "fit_max_round": fit_max,
                "development_round": development_round,
                "test_low_round": test_low,
                "test_high_round": test_high,
                "test_instances": int(np.sum(test)),
                "methods": methods,
                "strongest_baseline": strongest_baseline,
                "relative_mse_reduction_vs_strongest_baseline": float(
                    1.0 - methods["effective_toggle_persistence"]["mse"]
                    / methods[strongest_baseline]["mse"]
                ),
                "paired_win_fraction_vs_strongest_baseline": float(
                    np.mean(proposal_error < baseline_error)
                ),
                "group_count": len(fits),
                "stationary_grid_points": next(iter(stationary_fits.values())).grid_points,
                "test_rounds": rounds[test].tolist(),
                "test_target": target[test].tolist(),
                "test_predictions": {name: value[test].tolist() for name, value in predictions.items()},
            }
        )

    effective = np.asarray(
        [inverse_toggle_probability(float(value), int(depth)) for value, depth in zip(target, rounds)]
    )
    trajectory = []
    for depth in sorted(set(rounds.tolist())):
        values = effective[rounds == depth]
        trajectory.append(
            {
                "round": depth,
                "mean_effective_toggle": float(np.mean(values)),
                "minimum_effective_toggle": float(np.min(values)),
                "maximum_effective_toggle": float(np.max(values)),
            }
        )
    return {"endpoint": endpoint, "role": role, "folds": fold_records, "effective_toggle_trajectory": trajectory}


def _timing_study(rounds, groups, target) -> list[dict[str, Any]]:
    ordered_groups = list(dict.fromkeys(groups))
    records = []
    train = rounds <= 17
    for count in (2, 4, 6, 8, len(ordered_groups)):
        selected = set(ordered_groups[:count])
        mask = np.fromiter((group in selected for group in groups), dtype=bool, count=len(groups))
        local_rounds = rounds[mask]
        local_target = target[mask]
        local_groups = [group for group in groups if group in selected]
        local_train = train[mask]
        functions: dict[str, Callable[[], Any]] = {
            "effective_toggle_persistence": lambda: grouped_effective_toggle_persistence(local_rounds, local_target, local_groups, local_train),
            "last_observed_rate": lambda: grouped_last_rate(local_rounds, local_target, local_groups, local_train),
            "stationary_toggle_fit": lambda: grouped_toggle_forecast(local_rounds, local_target, local_groups, local_train),
        }
        method_timings = {}
        for name, function in functions.items():
            repeats = 5 if name == "stationary_toggle_fit" else 101
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                function()
                samples.append(time.perf_counter() - start)
            method_timings[name] = {
                "repeats": repeats,
                "median_seconds": float(np.median(samples)),
                "q1_seconds": float(np.quantile(samples, 0.25)),
                "q3_seconds": float(np.quantile(samples, 0.75)),
            }
        records.append({"groups": count, "rows": int(np.sum(mask)), "methods": method_timings})
    return records


def run_study(data_path: Path) -> dict[str, Any]:
    rows, rounds, groups, features = _load(data_path)
    endpoints = [
        _endpoint_study(name, role, rows, rounds, groups, features)
        for name, role in ENDPOINTS
    ]
    development_target = np.asarray(
        [float(row[ENDPOINTS[0][0]]) for row in rows], dtype=float
    )
    timing = _timing_study(rounds, groups, development_target)
    return {
        "schema_version": 1,
        "title": "Effective-toggle persistence on surface-code hardware records",
        "source": {
            "dataset": "Google Quantum error-correction hardware experiment summary",
            "path": str(data_path),
            "sha256": _sha256(data_path),
            "experiments": len(rows),
            "shots_per_experiment": 50_000,
            "bases": sorted(set(row["basis"] for row in rows)),
            "distances": sorted(set(int(row["distance"]) for row in rows)),
            "rounds": sorted(set(rounds.tolist())),
            "groups": len(set(groups)),
        },
        "protocol": {
            "development_endpoint": ENDPOINTS[0][0],
            "confirmation_endpoint": ENDPOINTS[1][0],
            "rolling_folds": [list(value) for value in FOLDS],
            "strong_baselines": ["last_observed_rate", "stationary_toggle_fit", "ridge_extrapolation"],
        },
        "endpoints": endpoints,
        "timing": timing,
        "platform": {"machine": platform.machine(), "system": platform.platform(), "python": platform.python_version(), "numpy": np.__version__},
        "checks": {
            "source_authenticated": _sha256(data_path) == SOURCE_SHA256,
            "proposal_wins_all_development_folds": all(
                fold["relative_mse_reduction_vs_strongest_baseline"] > 0.0
                for fold in endpoints[0]["folds"]
            ),
            "proposal_wins_all_confirmation_folds": all(
                fold["relative_mse_reduction_vs_strongest_baseline"] > 0.0
                for fold in endpoints[1]["folds"]
            ),
        },
    }


def semantic_sha256(results: dict[str, Any]) -> str:
    portable = json.loads(json.dumps(results))
    portable.pop("timing", None)
    portable.pop("platform", None)
    payload = json.dumps(portable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_outputs(results: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "persistence_study.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "rolling_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["endpoint", "role", "fold", "test_low_round", "test_high_round", "method", "mse", "mae", "maximum_absolute_error"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for endpoint in results["endpoints"]:
            for fold_index, fold in enumerate(endpoint["folds"], start=1):
                for method, metrics in fold["methods"].items():
                    writer.writerow({"endpoint": endpoint["endpoint"], "role": endpoint["role"], "fold": fold_index, "test_low_round": fold["test_low_round"], "test_high_round": fold["test_high_round"], "method": method, **metrics})
    _write_tex(results, output / "persistence_numbers.tex")
    _write_figures(results, output / "figures")


def _write_tex(results: dict[str, Any], path: Path) -> None:
    rows = []
    for endpoint in results["endpoints"]:
        for index, fold in enumerate(endpoint["folds"], start=1):
            proposed = fold["methods"]["effective_toggle_persistence"]["mse"]
            baseline = fold["methods"][fold["strongest_baseline"]]["mse"]
            rows.append(
                f"{endpoint['role']} & {index} & {fold['test_low_round']}--{fold['test_high_round']} & "
                f"{proposed:.2e} & {fold['strongest_baseline'].replace('_', ' ')} & {baseline:.2e} & "
                f"{100*fold['relative_mse_reduction_vs_strongest_baseline']:.1f}\\% \\\\"
            )
    path.write_text(
        "\\newcommand{\\PersistenceRows}{%\n" + "\n".join(rows) + "\n}\n"
        + f"\\newcommand{{\\PersistenceHash}}{{{semantic_sha256(results)}}}\n",
        encoding="utf-8",
    )


def _write_figures(results: dict[str, Any], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    colors = {"effective_toggle_persistence": "#1f4e79", "last_observed_rate": "#a23b2a", "stationary_toggle_fit": "#4f772d", "ridge_extrapolation": "#6a4c93"}
    labels = {"effective_toggle_persistence": "effective-toggle persistence", "last_observed_rate": "last rate", "stationary_toggle_fit": "stationary toggle", "ridge_extrapolation": "ridge"}

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), sharey=True)
    for axis, endpoint in zip(axes, results["endpoints"]):
        trajectory = endpoint["effective_toggle_trajectory"]
        x = np.asarray([row["round"] for row in trajectory])
        mean = np.asarray([row["mean_effective_toggle"] for row in trajectory])
        low = np.asarray([row["minimum_effective_toggle"] for row in trajectory])
        high = np.asarray([row["maximum_effective_toggle"] for row in trajectory])
        axis.plot(x, mean, "o-", color="#1f4e79")
        axis.fill_between(x, low, high, color="#1f4e79", alpha=0.18)
        axis.set_title(endpoint["role"])
        axis.set_xlabel("circuit rounds")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("effective per-round toggle")
    fig.tight_layout()
    fig.savefig(destination / "effective_toggle_trajectories.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(destination / "effective_toggle_trajectories.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), sharey=True)
    for axis, endpoint in zip(axes, results["endpoints"]):
        x = np.arange(1, len(endpoint["folds"]) + 1)
        for method in colors:
            y = [fold["methods"][method]["mse"] for fold in endpoint["folds"]]
            axis.semilogy(x, y, "o-", color=colors[method], label=labels[method])
        axis.set_title(endpoint["role"])
        axis.set_xlabel("rolling fold")
        axis.set_xticks(x)
        axis.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("held-out mean-squared error")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(destination / "rolling_error.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(destination / "rolling_error.png", dpi=220)
    plt.close(fig)

    confirmation = results["endpoints"][1]["folds"][-1]
    rounds = np.asarray(confirmation["test_rounds"])
    target = np.asarray(confirmation["test_target"])
    unique_rounds = sorted(set(rounds.tolist()))
    fig, axis = plt.subplots(figsize=(5.4, 3.2))
    axis.plot(unique_rounds, [target[rounds == r].mean() for r in unique_rounds], "ko-", label="measured")
    for method in ("effective_toggle_persistence", "last_observed_rate", "stationary_toggle_fit"):
        prediction = np.asarray(confirmation["test_predictions"][method])
        axis.plot(unique_rounds, [prediction[rounds == r].mean() for r in unique_rounds], "o-", color=colors[method], label=labels[method])
    axis.set_xlabel("circuit rounds")
    axis.set_ylabel("belief-matching logical error rate")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "confirmation_forecast.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(destination / "confirmation_forecast.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.4, 3.2))
    for method in ("effective_toggle_persistence", "last_observed_rate", "stationary_toggle_fit"):
        axis.semilogy(
            [row["groups"] for row in results["timing"]],
            [row["methods"][method]["median_seconds"] for row in results["timing"]],
            "o-", color=colors[method], label=labels[method],
        )
    axis.set_xlabel("hardware strata")
    axis.set_ylabel("median fit-and-predict time (s)")
    axis.grid(alpha=0.25, which="both")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "runtime_scaling.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(destination / "runtime_scaling.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2] / "data/source/google_qec3v5_experiment_summary.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = run_study(args.data)
    write_outputs(results, args.output)
    print(json.dumps({"checks": results["checks"], "semantic_sha256": semantic_sha256(results)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
