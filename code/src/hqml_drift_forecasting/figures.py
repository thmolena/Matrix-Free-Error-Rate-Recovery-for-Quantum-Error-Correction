from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import experiment
from .matrix_free import matrix_free_certificate
from .saturation import inverse_toggle_probability


COLORS = {
    "blue": "#1f77b4",
    "orange": "#e67e22",
    "green": "#16856b",
    "red": "#b23a48",
    "purple": "#6f4e9c",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(
        path,
        metadata={
            "Creator": "hqml-drift-forecasting 3.0.0",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def _hardware_rows() -> list[dict[str, str]]:
    path = (
        experiment.PACKAGE_ROOT
        / "data"
        / "source"
        / "google_qec3v5_experiment_summary.csv"
    )
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _figure_landscapes(rows: list[dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.35), sharey=True)
    for axis, basis in zip(axes, ("X", "Z")):
        for distance, color, marker in (
            (3, COLORS["blue"], "o"),
            (5, COLORS["orange"], "s"),
        ):
            subset = [
                row
                for row in rows
                if row["basis"] == basis and int(row["distance"]) == distance
            ]
            rounds = sorted({int(row["rounds"]) for row in subset})
            means = [
                np.mean(
                    [
                        float(row["tensor_network_error_rate"])
                        for row in subset
                        if int(row["rounds"]) == round_count
                    ]
                )
                for round_count in rounds
            ]
            axis.plot(
                rounds,
                means,
                color=color,
                marker=marker,
                linewidth=1.4,
                markersize=3.5,
                label=f"distance {distance}",
            )
        axis.axvline(17, color="#666666", linestyle="--", linewidth=0.9)
        axis.set_xlabel("surface-code rounds")
        axis.set_title(f"logical {basis}-basis experiments")
        axis.legend(frameon=False)
    axes[0].set_ylabel("tensor-network decoder error rate")
    _save(fig, output / "hardware_error_landscapes.pdf")


def _figure_model_bars(result: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.35))
    labels = ["linear\nridge", "legacy\nDM-RCL", "toggle\nbaseline"]
    values = [
        result["hardware_baseline_mse"],
        result["hardware_learned_mse"],
        result["hardware_toggle_mse"],
    ]
    bars = axes[0].bar(
        labels,
        values,
        color=[COLORS["blue"], COLORS["green"], COLORS["orange"]],
        width=0.62,
    )
    axes[0].set_ylabel("held-out mean-squared error")
    axes[0].set_title("final rolling-origin split")
    for bar, value in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2e}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    reductions = 100 * np.asarray(
        result["hardware_fold_toggle_over_legacy_reductions"]
    )
    bars = axes[1].bar(
        ["13–15", "15–17", "17–19", "19–25"],
        reductions,
        color=COLORS["purple"],
        width=0.65,
    )
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_ylabel("toggle reduction vs legacy DM-RCL (%)")
    axes[1].set_xlabel("held-out round interval")
    axes[1].set_title("four chronological folds")
    for bar, value in zip(bars, reductions):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    _save(fig, output / "hardware_recovery_bars.pdf")


def _folds() -> list[dict[str, object]]:
    raw, target, rounds, _ = experiment._hardware_arrays()
    specs = [(9, 11, 13, 15), (11, 13, 15, 17), (13, 15, 17, 19), (15, 17, 19, 25)]
    return [
        experiment._rolling_hardware_fold(
            raw,
            target,
            rounds,
            fit_max=fit_max,
            development_round=development,
            test_low=test_low,
            test_high=test_high,
            seed=experiment.CONFIG["seed"] + fit_max,
        )
        for fit_max, development, test_low, test_high in specs
    ]


def _figure_rolling_lines(folds: list[dict[str, float | int]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.35))
    x = np.arange(1, len(folds) + 1)
    axes[0].plot(
        x,
        [fold["baseline_mse"] for fold in folds],
        marker="o",
        color=COLORS["blue"],
        label="linear ridge",
    )
    axes[0].plot(
        x,
        [fold["learned_mse"] for fold in folds],
        marker="s",
        color=COLORS["green"],
        label="legacy DM-RCL",
    )
    axes[0].plot(
        x,
        [fold["toggle_mse"] for fold in folds],
        marker="^",
        color=COLORS["orange"],
        label="toggle baseline",
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xlabel("rolling-origin fold")
    axes[0].set_ylabel("held-out MSE (log scale)")
    axes[0].legend(frameon=False)
    axes[0].set_title("prediction error")
    axes[1].plot(
        x,
        [fold["selected_alpha"] for fold in folds],
        marker="o",
        color=COLORS["purple"],
        label=r"legacy $\alpha$",
    )
    axes[1].plot(
        x,
        [fold["audited_selected_alpha"] for fold in folds],
        marker="s",
        color=COLORS["red"],
        label=r"audited $\alpha$",
    )
    axes[1].set_xticks(x)
    axes[1].set_xlabel("rolling-origin fold")
    axes[1].set_ylabel("selected matrix-free weight")
    axes[1].set_ylim(-0.01, 0.22)
    axes[1].legend(frameon=False)
    axes[1].set_title("weak-baseline versus audited selection")
    _save(fig, output / "rolling_origin_lines.pdf")


def _figure_toggle_diagnostics(
    rows: list[dict[str, str]], output: Path
) -> None:
    """Plot per-depth inferred toggle rates and simultaneous shot intervals."""

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), sharey=True)
    for axis, basis in zip(axes, ("X", "Z")):
        subset = [row for row in rows if row["basis"] == basis and int(row["rounds"]) <= 17]
        keys = list(
            dict.fromkeys(
                (row["distance"], row["center_row"], row["center_col"])
                for row in subset
            )
        )
        for index, key in enumerate(keys):
            group = [
                row
                for row in subset
                if (row["distance"], row["center_row"], row["center_col"]) == key
            ]
            depth = np.asarray([int(row["rounds"]) for row in group], dtype=int)
            rate = np.asarray(
                [float(row["tensor_network_error_rate"]) for row in group]
            )
            shots = np.asarray([int(row["shots"]) for row in group], dtype=int)
            epsilon = np.sqrt(
                np.log(2.0 * len(depth) / 0.005) / (2.0 * shots)
            )
            center = np.asarray(
                [inverse_toggle_probability(p, int(r)) for p, r in zip(rate, depth)]
            )
            lower = np.asarray(
                [
                    inverse_toggle_probability(max(0.0, p - e), int(r))
                    for p, e, r in zip(rate, epsilon, depth)
                ]
            )
            upper = np.asarray(
                [
                    inverse_toggle_probability(min(0.5, p + e), int(r))
                    for p, e, r in zip(rate, epsilon, depth)
                ]
            )
            label = f"d={key[0]}, center=({key[1]},{key[2]})"
            axis.errorbar(
                depth,
                center,
                yerr=np.vstack((center - lower, upper - center)),
                marker=("o", "s", "^", "D", "v")[index % 5],
                markersize=2.6,
                linewidth=0.8,
                capsize=1.2,
                label=label,
            )
        axis.set_xlabel("surface-code rounds")
        axis.set_title(f"logical {basis} basis")
        axis.legend(frameon=False, fontsize=5.7, ncol=1)
    axes[0].set_ylabel("rate inferred under stationary-toggle model")
    _save(fig, output / "toggle_model_diagnostic.pdf")


def _rank_sweep() -> list[dict[str, float | int | bool]]:
    raw, _, _, _ = experiment._hardware_arrays()
    output = []
    for rank in (2, 4, 6, 8, 12):
        metrics = matrix_free_certificate(
            raw,
            experiment.CONFIG["seed"],
            bandwidth=0.40,
            leaf_size=16,
            low_rank=2,
            high_rank=rank,
        )
        output.append({"rank": rank, **metrics})
    return output


def _figure_rank_lines(sweep: list[dict], output: Path) -> None:
    ranks = [row["rank"] for row in sweep]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.25))
    axes[0].plot(
        ranks,
        [row["hmat_sampled_row_relative_error"] for row in sweep],
        marker="o",
        color=COLORS["red"],
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("sampled-row relative error")
    axes[1].plot(
        ranks,
        [100 * row["hmat_storage_fraction"] for row in sweep],
        marker="s",
        color=COLORS["green"],
    )
    axes[1].set_ylabel("dense storage (%)")
    axes[2].plot(
        ranks,
        [row["hmat_matmat_work_ratio"] for row in sweep],
        marker="^",
        color=COLORS["blue"],
    )
    axes[2].axhline(1, color="#666666", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("dense/structured work")
    for axis in axes:
        axis.set_xlabel("hierarchical rank")
    _save(fig, output / "rank_certificate_lines.pdf")


def _figure_scale_bars(result: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.35))
    labels = ["hardware\n130 rows", "stress\n75,598 rows"]
    storage = [
        100 * result["hmat_storage_fraction"],
        100 * result["stress_hmat_storage_fraction"],
    ]
    work = [
        result["hmat_matmat_work_ratio"],
        result["stress_hmat_matmat_work_ratio"],
    ]
    axes[0].bar(labels, storage, color=[COLORS["orange"], COLORS["green"]], width=0.6)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("stored scalars / dense (%)")
    axes[0].set_title("matrix-free storage")
    axes[1].bar(labels, work, color=[COLORS["orange"], COLORS["blue"]], width=0.6)
    axes[1].set_yscale("log")
    axes[1].axhline(1, color="#666666", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("dense/structured counted work")
    axes[1].set_title("batched application")
    _save(fig, output / "scale_certificate_bars.pdf")


def _write_tables(
    rows: list[dict[str, str]],
    folds: list[dict[str, object]],
    result: dict,
    output: Path,
) -> None:
    grouped = []
    for basis in ("X", "Z"):
        for distance in (3, 5):
            subset = [
                row
                for row in rows
                if row["basis"] == basis and int(row["distance"]) == distance
            ]
            target = [float(row["tensor_network_error_rate"]) for row in subset]
            grouped.append(
                f"{basis} & {distance} & {len(subset)} & "
                f"{sum(int(row['shots']) for row in subset):,} & "
                f"{min(target):.5f} & {max(target):.5f} \\\\"
            )
    (output / "hardware_data.tex").write_text(
        "\n".join(grouped) + "\n\\bottomrule\n"
    )
    fold_lines = [
        f"{index} & {fold['development_round']} & "
        f"{fold['test_low_round']}--{fold['test_high_round']} & "
        f"{fold['test_instances']} & {fold['baseline_mse']:.3e} & "
        f"{fold['learned_mse']:.3e} & {fold['toggle_mse']:.3e} & "
        f"{fold['audited_selected_alpha']:.3f} & "
        f"{100 * fold['toggle_over_legacy_relative_mse_reduction']:.1f} \\\\"
        for index, fold in enumerate(folds, start=1)
    ]
    (output / "rolling_folds.tex").write_text(
        "\n".join(fold_lines) + "\n\\bottomrule\n"
    )
    stress_lines = [
        f"Hardware primary & {result['hardware_experiments']:,} & "
        f"{100 * result['hmat_storage_fraction']:.3f} & "
        f"{result['hmat_matmat_work_ratio']:.2f} & "
        f"{result['hmat_sampled_row_relative_error']:.2e} \\\\",
        f"Simulated stress & {result['stress_unique_weighted_rows']:,} & "
        f"{100 * result['stress_hmat_storage_fraction']:.3f} & "
        f"{result['stress_hmat_matmat_work_ratio']:.1f} & "
        f"{result['stress_hmat_sampled_row_relative_error']:.2e} \\\\",
    ]
    (output / "scale_comparison.tex").write_text(
        "\n".join(stress_lines) + "\n\\bottomrule\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the manuscript's line graphs, bar charts, and tables"
    )
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    _style()
    rows = _hardware_rows()
    result = json.loads(
        (experiment.PACKAGE_ROOT / "results" / "results.json").read_text()
    )
    folds = _folds()
    sweep = _rank_sweep()
    _figure_landscapes(rows, args.figure_dir)
    _figure_model_bars(result, args.figure_dir)
    _figure_rolling_lines(folds, args.figure_dir)
    _figure_toggle_diagnostics(rows, args.figure_dir)
    _figure_rank_lines(sweep, args.figure_dir)
    _figure_scale_bars(result, args.figure_dir)
    _write_tables(rows, folds, result, args.table_dir)
    print(
        f"wrote 6 allowed figures to {args.figure_dir} "
        f"and 3 numeric tables to {args.table_dir}"
    )


if __name__ == "__main__":
    main()
