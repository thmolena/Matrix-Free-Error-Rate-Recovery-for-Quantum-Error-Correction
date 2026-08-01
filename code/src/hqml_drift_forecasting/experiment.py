from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from .matrix_free import (
    hierarchical_prior_bundle,
    matrix_free_certificate,
    select_rank_counterfactual_blend,
)
from .saturation import grouped_toggle_forecast, simultaneous_toggle_interval

SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT if (PROJECT_ROOT / "results").is_dir() else SOURCE_ROOT
CONFIG = {
    "key": "drift_forecasting",
    "main": "../main.tex",
    "title": "A Saturation-Baseline Audit of Matrix-Free Quantum Error-Rate Recovery",
    "domain": "qec",
    "task": "hardware surface-code decoder-error recovery",
    "question": "Does the legacy matrix-free correction improve a standard saturation baseline on frozen held-out hardware rounds?",
    "mechanism": "A groupwise binary-toggle saturation curve is fitted before holdout evaluation; the legacy hierarchical residual is retained as a falsified comparator and in a separate scale audit.",
    "loss": "held-out mean-squared error with development-only selection; simultaneous finite-shot intervals separately test stationary-toggle compatibility",
    "seed": 1303,
    "split": "rolling circuit depth",
    "package": "hqml_drift_forecasting",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ridge_fit(design: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    penalty = 0.01 * np.eye(design.shape[1])
    return np.linalg.solve(
        design[mask].T @ design[mask] + penalty,
        design[mask].T @ target[mask],
    )


def _hardware_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    path = PACKAGE_ROOT / "data" / "source" / "google_qec3v5_experiment_summary.csv"
    expected = "15ffb8c7773a4fa5d1d498bbe01fd929334c0c13da69ab81704ad13beadde5f5"
    if sha256(path) != expected:
        raise ValueError("Google hardware summary SHA-256 mismatch")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    features = np.asarray(
        [
            [
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
    target = np.asarray(
        [float(row["tensor_network_error_rate"]) for row in rows], dtype=float
    )
    rounds = np.asarray([int(row["rounds"]) for row in rows], dtype=int)
    return features, target, rounds, rows


def _rolling_hardware_fold(
    raw_features: np.ndarray,
    target: np.ndarray,
    rounds: np.ndarray,
    *,
    fit_max: int,
    development_round: int,
    test_low: int,
    test_high: int,
    seed: int,
) -> dict[str, object]:
    fit = rounds <= fit_max
    development = rounds == development_round
    train = rounds <= development_round
    test = (rounds >= test_low) & (rounds <= test_high)
    # A standard independent-toggle saturation law is a necessary physical
    # baseline for repeated logical-error probabilities.  Groups exclude
    # circuit depth and use only basis, distance, and patch location.
    groups = [
        tuple(row)
        for row in raw_features[:, np.asarray([0, 1, 3, 4], dtype=int)]
    ]
    development_toggle, development_toggle_fits = grouped_toggle_forecast(
        rounds, target, groups, fit
    )
    test_toggle, test_toggle_fits = grouped_toggle_forecast(
        rounds, target, groups, train
    )
    mean = raw_features[fit].mean(axis=0)
    std = raw_features[fit].std(axis=0)
    std[std < 1e-9] = 1.0
    standardized = (raw_features - mean) / std
    design = np.column_stack((np.ones(len(standardized)), standardized))

    development_baseline = design[development] @ _ridge_fit(
        design, target, fit
    )
    test_baseline = design[test] @ _ridge_fit(design, target, train)
    prior = hierarchical_prior_bundle(
        standardized,
        [fit, train],
        target,
        seed,
        bandwidth=0.40,
        leaf_size=16,
        low_rank=2,
        high_rank=8,
    )
    development_prior_low = prior["low"][0][development, 0]
    development_prior_high = prior["high"][0][development, 0]
    monotone_reference = []
    monotone_baseline = []
    monotone_toggle = []
    monotone_prior_high = []
    development_features = raw_features[development]
    for basis in (0.0, 1.0):
        for distance_five in (0.0, 1.0):
            reference_group = (
                (rounds == fit_max)
                & (raw_features[:, 0] == basis)
                & (raw_features[:, 1] == distance_five)
            )
            development_group = (
                (development_features[:, 0] == basis)
                & (development_features[:, 1] == distance_five)
            )
            if reference_group.any() and development_group.any():
                monotone_reference.append(float(target[reference_group].mean()))
                monotone_baseline.append(
                    float(development_baseline[development_group].mean())
                )
                monotone_toggle.append(
                    float(development_toggle[development][development_group].mean())
                )
                monotone_prior_high.append(
                    float(development_prior_high[development_group].mean())
                )
    monotone_reference_array = np.asarray(monotone_reference, dtype=float)
    monotone_baseline_array = np.asarray(monotone_baseline, dtype=float)
    monotone_toggle_array = np.asarray(monotone_toggle, dtype=float)
    monotone_prior_high_array = np.asarray(monotone_prior_high, dtype=float)
    selected = select_rank_counterfactual_blend(
        target[development],
        development_baseline,
        development_prior_low,
        development_prior_high,
        loss="mse",
        sampled_operator_error=prior["metrics"][
            "hmat_sampled_row_relative_error"
        ],
        storage_fraction=prior["metrics"]["hmat_storage_fraction"],
        monotone_reference=monotone_reference_array,
        monotone_baseline=monotone_baseline_array,
        monotone_prior_high=monotone_prior_high_array,
    )
    # Repeat the identical selection protocol with the physically standard
    # saturation curve in place of the weak linear extrapolator.  This is the
    # decisive baseline audit: the matrix-free correction may abstain by
    # selecting alpha=0, and held-out labels remain unavailable to selection.
    selected_toggle = select_rank_counterfactual_blend(
        target[development],
        development_toggle[development],
        development_prior_low,
        development_prior_high,
        loss="mse",
        sampled_operator_error=prior["metrics"][
            "hmat_sampled_row_relative_error"
        ],
        storage_fraction=prior["metrics"]["hmat_storage_fraction"],
        monotone_reference=monotone_reference_array,
        monotone_baseline=monotone_toggle_array,
        monotone_prior_high=monotone_prior_high_array,
    )
    alpha = selected["alpha"]
    prediction = np.clip(
        (1.0 - alpha) * test_baseline
        + alpha * prior["high"][1][test, 0],
        0.0,
        1.0,
    )
    prediction_low = np.clip(
        (1.0 - alpha) * test_baseline
        + alpha * prior["low"][1][test, 0],
        0.0,
        1.0,
    )
    audited_alpha = selected_toggle["alpha"]
    audited_prediction = np.clip(
        (1.0 - audited_alpha) * test_toggle[test]
        + audited_alpha * prior["high"][1][test, 0],
        0.0,
        1.0,
    )
    audited_prediction_low = np.clip(
        (1.0 - audited_alpha) * test_toggle[test]
        + audited_alpha * prior["low"][1][test, 0],
        0.0,
        1.0,
    )
    baseline_mse = float(np.mean((test_baseline - target[test]) ** 2))
    learned_mse = float(np.mean((prediction - target[test]) ** 2))
    baseline_squared_error = (test_baseline - target[test]) ** 2
    learned_squared_error = (prediction - target[test]) ** 2
    toggle_squared_error = (test_toggle[test] - target[test]) ** 2
    audited_squared_error = (audited_prediction - target[test]) ** 2
    rng = np.random.default_rng(seed + 104729)
    bootstrap_indices = rng.integers(
        0, len(learned_squared_error), size=(10_000, len(learned_squared_error))
    )
    bootstrap_baseline = baseline_squared_error[bootstrap_indices].mean(axis=1)
    bootstrap_learned = learned_squared_error[bootstrap_indices].mean(axis=1)
    bootstrap_reduction = 1.0 - bootstrap_learned / np.maximum(
        bootstrap_baseline, 1e-15
    )
    bootstrap_toggle = toggle_squared_error[bootstrap_indices].mean(axis=1)
    bootstrap_toggle_over_legacy = 1.0 - bootstrap_toggle / np.maximum(
        bootstrap_learned, 1e-15
    )
    target_scale = max(float(np.var(target[test])), 1e-12)
    rank_counterfactual = float(
        np.mean((prediction - prediction_low) ** 2) / target_scale
    )
    audited_rank_counterfactual = float(
        np.mean((audited_prediction - audited_prediction_low) ** 2)
        / target_scale
    )
    compatibility = []
    unique_groups = list(dict.fromkeys(groups))
    for group in unique_groups:
        in_group = np.fromiter(
            (item == group for item in groups),
            dtype=bool,
            count=len(rounds),
        )
        known_group = in_group & train
        compatibility.append(
            simultaneous_toggle_interval(
                rounds[known_group],
                target[known_group],
                np.full(int(known_group.sum()), 50_000, dtype=int),
                delta=0.05 / len(unique_groups),
            ).compatible
        )
    return {
        "fit_max_round": fit_max,
        "development_round": development_round,
        "test_low_round": test_low,
        "test_high_round": test_high,
        "test_instances": int(test.sum()),
        "selected_alpha": float(alpha),
        "baseline_mse": baseline_mse,
        "learned_mse": learned_mse,
        "learned_mae": float(np.mean(np.abs(prediction - target[test]))),
        "relative_mse_reduction": float(
            1.0 - learned_mse / max(baseline_mse, 1e-15)
        ),
        "paired_win_fraction": float(
            np.mean(learned_squared_error < baseline_squared_error)
        ),
        "relative_mse_reduction_ci_low": float(
            np.quantile(bootstrap_reduction, 0.025)
        ),
        "relative_mse_reduction_ci_high": float(
            np.quantile(bootstrap_reduction, 0.975)
        ),
        "toggle_mse": float(np.mean(toggle_squared_error)),
        "toggle_mae": float(np.mean(np.abs(test_toggle[test] - target[test]))),
        "toggle_grid_points": int(
            next(iter(test_toggle_fits.values())).grid_points
        ),
        "toggle_max_grid_objective_gap_bound": float(
            max(fit.grid_objective_gap_bound for fit in test_toggle_fits.values())
        ),
        "toggle_compatible_groups": int(sum(compatibility)),
        "toggle_total_groups": int(len(compatibility)),
        "toggle_familywise_delta": 0.05,
        "toggle_over_legacy_relative_mse_reduction": float(
            1.0 - np.mean(toggle_squared_error) / max(learned_mse, 1e-15)
        ),
        "toggle_over_legacy_ci_low": float(
            np.quantile(bootstrap_toggle_over_legacy, 0.025)
        ),
        "toggle_over_legacy_ci_high": float(
            np.quantile(bootstrap_toggle_over_legacy, 0.975)
        ),
        "toggle_over_legacy_paired_win_fraction": float(
            np.mean(toggle_squared_error < learned_squared_error)
        ),
        "audited_selected_alpha": float(audited_alpha),
        "audited_mse": float(np.mean(audited_squared_error)),
        "audited_mae": float(
            np.mean(np.abs(audited_prediction - target[test]))
        ),
        "audited_rank_counterfactual_term": audited_rank_counterfactual,
        "audited_development_total_loss": selected_toggle["total_loss"],
        "rank_counterfactual_term": rank_counterfactual,
        "depth_monotonicity_term": selected["monotonicity_term"],
        "depth_monotonicity_weight": selected["monotonicity_weight"],
        "development_total_loss": selected["total_loss"],
        "diagnostic_test_indices": np.flatnonzero(test).tolist(),
        "diagnostic_test_target": target[test].tolist(),
        "diagnostic_test_baseline": test_baseline.tolist(),
        "diagnostic_test_prediction": prediction.tolist(),
        "diagnostic_test_toggle": test_toggle[test].tolist(),
        "diagnostic_test_audited": audited_prediction.tolist(),
        "diagnostic_development_target": target[development].tolist(),
        "diagnostic_development_baseline": development_baseline.tolist(),
        "diagnostic_development_toggle": development_toggle[development].tolist(),
        "diagnostic_development_prior_low": prior["low"][0][
            development, 0
        ].tolist(),
        "diagnostic_development_prior_high": prior["high"][0][
            development, 0
        ].tolist(),
        "diagnostic_monotone_reference": monotone_reference,
        "diagnostic_monotone_baseline": monotone_baseline,
        "diagnostic_monotone_toggle": monotone_toggle,
        "diagnostic_monotone_prior_high": monotone_prior_high,
        **prior["metrics"],
    }


def run_hardware_qec() -> dict:
    raw_features, target, rounds, rows = _hardware_arrays()
    fold_specs = [
        (9, 11, 13, 15),
        (11, 13, 15, 17),
        (13, 15, 17, 19),
        (15, 17, 19, 25),
    ]
    folds = [
        _rolling_hardware_fold(
            raw_features,
            target,
            rounds,
            fit_max=fit_max,
            development_round=development_round,
            test_low=test_low,
            test_high=test_high,
            seed=CONFIG["seed"] + fit_max,
        )
        for fit_max, development_round, test_low, test_high in fold_specs
    ]
    final = folds[-1]
    reductions = np.asarray(
        [fold["relative_mse_reduction"] for fold in folds], dtype=float
    )
    return {
        "hardware_dataset": "Google Quantum AI surface-code hardware experiments",
        "hardware_dataset_url": "https://doi.org/10.5281/zenodo.6804040",
        "hardware_archive_sha256": "ef569742126b305e163cc7fccbf6275e03bdd54d090dffb1aef12a78af0a74c4",
        "hardware_summary_sha256": "15ffb8c7773a4fa5d1d498bbe01fd929334c0c13da69ab81704ad13beadde5f5",
        "hardware_license": "CC BY 4.0",
        "hardware_experiments": len(rows),
        "hardware_shots": int(sum(int(row["shots"]) for row in rows)),
        "hardware_distances": 2,
        "hardware_round_settings": len(np.unique(rounds)),
        "hardware_final_train_instances": int((rounds <= 17).sum()),
        "hardware_final_test_instances": int((rounds >= 19).sum()),
        "hardware_baseline_mse": final["baseline_mse"],
        "hardware_learned_mse": final["learned_mse"],
        "hardware_learned_mae": final["learned_mae"],
        "hardware_relative_mse_reduction": final["relative_mse_reduction"],
        "hardware_paired_win_fraction": final["paired_win_fraction"],
        "hardware_relative_mse_reduction_ci_low": final[
            "relative_mse_reduction_ci_low"
        ],
        "hardware_relative_mse_reduction_ci_high": final[
            "relative_mse_reduction_ci_high"
        ],
        "hardware_selected_alpha": final["selected_alpha"],
        "hardware_rank_counterfactual_term": final["rank_counterfactual_term"],
        "hardware_depth_monotonicity_term": final[
            "depth_monotonicity_term"
        ],
        "hardware_depth_monotonicity_weight": final[
            "depth_monotonicity_weight"
        ],
        "hardware_fold_depth_monotonicity_terms": [
            fold["depth_monotonicity_term"] for fold in folds
        ],
        "hardware_fold_reductions": [
            fold["relative_mse_reduction"] for fold in folds
        ],
        "hardware_fold_alphas": [fold["selected_alpha"] for fold in folds],
        "hardware_fold_toggle_mses": [fold["toggle_mse"] for fold in folds],
        "hardware_fold_legacy_mses": [fold["learned_mse"] for fold in folds],
        "hardware_fold_audited_alphas": [
            fold["audited_selected_alpha"] for fold in folds
        ],
        "hardware_fold_toggle_over_legacy_reductions": [
            fold["toggle_over_legacy_relative_mse_reduction"] for fold in folds
        ],
        "hardware_toggle_mse": final["toggle_mse"],
        "hardware_toggle_mae": final["toggle_mae"],
        "hardware_toggle_grid_points": final["toggle_grid_points"],
        "hardware_toggle_max_grid_objective_gap_bound": final[
            "toggle_max_grid_objective_gap_bound"
        ],
        "hardware_toggle_compatible_groups": final[
            "toggle_compatible_groups"
        ],
        "hardware_toggle_total_groups": final["toggle_total_groups"],
        "hardware_toggle_familywise_delta": final[
            "toggle_familywise_delta"
        ],
        "hardware_toggle_over_legacy_relative_mse_reduction": final[
            "toggle_over_legacy_relative_mse_reduction"
        ],
        "hardware_toggle_over_legacy_ci_low": final[
            "toggle_over_legacy_ci_low"
        ],
        "hardware_toggle_over_legacy_ci_high": final[
            "toggle_over_legacy_ci_high"
        ],
        "hardware_toggle_over_legacy_paired_win_fraction": final[
            "toggle_over_legacy_paired_win_fraction"
        ],
        "hardware_audited_selected_alpha": final["audited_selected_alpha"],
        "hardware_audited_mse": final["audited_mse"],
        "hardware_audited_mae": final["audited_mae"],
        "hardware_audited_rank_counterfactual_term": final[
            "audited_rank_counterfactual_term"
        ],
        "hardware_min_fold_reduction": float(reductions.min()),
        "hardware_median_fold_reduction": float(np.median(reductions)),
        "matrix_free": final["matrix_free"],
        "global_dense_matrix_materialized": final[
            "global_dense_matrix_materialized"
        ],
        "hmat_rank_low": final["hmat_rank_low"],
        "hmat_rank_high": final["hmat_rank_high"],
        "hmat_sampled_row_relative_error": final[
            "hmat_sampled_row_relative_error"
        ],
        "hmat_storage_fraction": final["hmat_storage_fraction"],
        "hmat_matmat_work_ratio": final["hmat_matmat_work_ratio"],
        "hmat_entry_query_fraction": final["hmat_entry_query_fraction"],
        "hmat_peak_block_fraction": final["hmat_peak_block_fraction"],
        "butterfly_inverse_error": final["butterfly_inverse_error"],
        "butterfly_work_ratio": final["butterfly_work_ratio"],
    }


def syndrome_features(value: str) -> np.ndarray:
    arr = np.asarray(ast.literal_eval(value), dtype=float).reshape(4, 4)
    flat = arr.ravel()
    temporal = np.abs(np.diff(arr, axis=0)).ravel()
    round_parity = np.mod(arr.sum(axis=1), 2)
    spatial_parity = np.mod(arr.sum(axis=0), 2)
    return np.concatenate((flat, temporal, round_parity, spatial_parity, [flat.sum()]))


def balanced_weights(y: np.ndarray, quantity: np.ndarray) -> np.ndarray:
    out = quantity.astype(float).copy()
    for label in (0.0, 1.0):
        mask = y == label
        out[mask] *= 0.5 / max(out[mask].sum(), 1.0)
    return out


def logistic_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    eye = np.eye(x.shape[1])
    eye[0, 0] = 0.0
    for _ in range(30):
        score = np.clip(x @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-score))
        grad = x.T @ (weights * (p - y)) + 1e-4 * (eye @ beta)
        curvature = weights * p * (1 - p)
        hess = (x.T * curvature) @ x + 1e-4 * eye + 1e-9 * np.eye(x.shape[1])
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.linalg.norm(step) < 1e-10:
            break
    return beta


def run_simulated_qec_stress() -> dict:
    path = PACKAGE_ROOT / "data" / "source" / "syndromes_dataset.zip"
    expected = "bdfce36a71f04295ac78fb372d9c2e381801c05e3be119f919750ef59026d072"
    if sha256(path) != expected:
        raise ValueError("QEC dataset SHA-256 mismatch")
    rows = []
    rates = []
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".csv"))
        for rate_index, name in enumerate(names):
            with archive.open(name) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
                for row in reader:
                    rows.append(
                        (
                            syndrome_features(row["syndromes"]),
                            float(row["labels"]),
                            float(row["quantity"]),
                        )
                    )
                    rates.append(rate_index)
    x = np.vstack([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    quantity = np.array([r[2] for r in rows])
    rates = np.array(rates)
    split = CONFIG.get("split", "rate")
    indices = np.arange(len(x))
    if split == "hash":
        test_mask = indices % 5 == 0
    elif split == "alternating":
        test_mask = rates % 2 == 1
    elif split == "middle":
        test_mask = rates == 4
    else:
        test_mask = rates >= 5
    train_mask = ~test_mask
    mean = np.average(x[train_mask], axis=0, weights=quantity[train_mask])
    var = np.average((x[train_mask] - mean) ** 2, axis=0, weights=quantity[train_mask])
    std = np.sqrt(var)
    std[std < 1e-8] = 1.0
    z = (x - mean) / std
    design = np.column_stack((np.ones(len(z)), z))
    train_rates = np.unique(rates[train_mask])
    development_rate = int(train_rates[-1])
    development_mask = train_mask & (rates == development_rate)
    fit_mask = train_mask & ~development_mask
    fit_weights = balanced_weights(y[fit_mask], quantity[fit_mask])
    beta_fit = logistic_fit(design[fit_mask], y[fit_mask], fit_weights)
    base_development = 1.0 / (
        1.0
        + np.exp(
            -np.clip(design[development_mask] @ beta_fit, -30, 30)
        )
    )
    train_weights = balanced_weights(y[train_mask], quantity[train_mask])
    beta = logistic_fit(design[train_mask], y[train_mask], train_weights)
    test_weights = balanced_weights(y[test_mask], quantity[test_mask])
    base_test = 1.0 / (
        1.0 + np.exp(-np.clip(design[test_mask] @ beta, -30, 30))
    )
    rate_feature = rates[:, None] / max(float(rates.max()), 1.0)
    prior = hierarchical_prior_bundle(
        np.column_stack((z, rate_feature)),
        [fit_mask, train_mask],
        y,
        CONFIG["seed"],
        bandwidth=0.10,
        leaf_size=64,
        low_rank=2,
        high_rank=8,
    )
    development_weights = balanced_weights(
        y[development_mask], quantity[development_mask]
    )
    selected = select_rank_counterfactual_blend(
        y[development_mask],
        base_development,
        np.clip(prior["low"][0][development_mask, 0], 1e-8, 1.0 - 1e-8),
        np.clip(prior["high"][0][development_mask, 0], 1e-8, 1.0 - 1e-8),
        weights=development_weights,
        loss="log",
        sampled_operator_error=prior["metrics"][
            "hmat_sampled_row_relative_error"
        ],
        storage_fraction=prior["metrics"]["hmat_storage_fraction"],
    )
    alpha = selected["alpha"]
    p = (1.0 - alpha) * base_test + alpha * np.clip(
        prior["high"][1][test_mask, 0], 1e-8, 1.0 - 1e-8
    )
    p_low = (1.0 - alpha) * base_test + alpha * np.clip(
        prior["low"][1][test_mask, 0], 1e-8, 1.0 - 1e-8
    )
    yt = y[test_mask]
    eps = 1e-12
    logistic_only_loss = float(
        np.sum(
            test_weights
            * (
                -(
                    yt * np.log(base_test + eps)
                    + (1 - yt) * np.log(1 - base_test + eps)
                )
            )
        )
    )
    learned_loss = float(
        np.sum(test_weights * (-(yt * np.log(p + eps) + (1 - yt) * np.log(1 - p + eps))))
    )
    baseline_loss = float(-math.log(0.5))
    logistic_only_brier = float(np.sum(test_weights * (base_test - yt) ** 2))
    learned_brier = float(np.sum(test_weights * (p - yt) ** 2))
    baseline_brier = 0.25
    accuracy = float(np.sum(test_weights * ((p >= 0.5) == yt)))
    target_scale = max(
        float(np.sum(test_weights * (yt - np.sum(test_weights * yt)) ** 2)),
        1e-8,
    )
    rank_counterfactual_term = float(
        np.sum(test_weights * (p - p_low) ** 2) / target_scale
    )
    rank_counterfactual_test_loss = (
        learned_loss
        + selected["consistency_weight"] * rank_counterfactual_term
        + selected["certificate_weight"] * selected["certificate_violation"]
        + selected["budget_weight"] * selected["budget_violation"]
    )
    return {
        "dataset": "Open Zenodo surface-code syndrome counts (distance 3)",
        "dataset_url": "https://doi.org/10.5281/zenodo.11166209",
        "dataset_sha256": expected,
        "records": int(quantity.sum()),
        "unique_weighted_rows": len(rows),
        "noise_rates": len(np.unique(rates)),
        "train_records": int(quantity[train_mask].sum()),
        "test_records": int(quantity[test_mask].sum()),
        "feature_dimension": design.shape[1],
        "baseline_loss": baseline_loss,
        "logistic_only_loss": logistic_only_loss,
        "learned_loss": learned_loss,
        "rank_selected_alpha": alpha,
        "rank_consistency_weight": selected["consistency_weight"],
        "rank_development_data_loss": selected["data_loss"],
        "rank_development_rank_counterfactual_term": selected[
            "rank_counterfactual_term"
        ],
        "rank_certificate_violation": selected["certificate_violation"],
        "rank_budget_violation": selected["budget_violation"],
        "rank_error_tolerance": selected["error_tolerance"],
        "rank_storage_budget": selected["storage_budget"],
        "rank_development_total_loss": selected["total_loss"],
        "rank_test_rank_counterfactual_term": rank_counterfactual_term,
        "rank_test_total_loss": rank_counterfactual_test_loss,
        "baseline_task_error": baseline_brier,
        "logistic_only_task_error": logistic_only_brier,
        "learned_task_error": learned_brier,
        "balanced_accuracy": accuracy,
        **prior["metrics"],
    }


def reproduce() -> dict:
    hardware = run_hardware_qec()
    stress = run_simulated_qec_stress()
    return {
        "schema_version": 3,
        "package": CONFIG["package"],
        "manuscript_key": CONFIG["key"],
        "seed": CONFIG["seed"],
        "task": CONFIG["task"],
        **hardware,
        **{f"stress_{key}": value for key, value in stress.items()},
    }


def _round_value(value):
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, list):
        return [_round_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _round_value(item) for key, item in value.items()}
    return value


def rounded(result: dict) -> dict:
    return {key: _round_value(value) for key, value in result.items()}


def tex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def write_outputs(result: dict) -> None:
    result = rounded(result)
    output = PACKAGE_ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    commands = {
        "DatasetName": tex_escape(str(result["hardware_dataset"])),
        "HardwareExperiments": f'{int(result["hardware_experiments"]):,}',
        "HardwareShots": f'{int(result["hardware_shots"]):,}',
        "HardwareTrain": str(result["hardware_final_train_instances"]),
        "HardwareTest": str(result["hardware_final_test_instances"]),
        "BaselineMSE": f'{result["hardware_baseline_mse"]:.6f}',
        "LearnedMSE": f'{result["hardware_learned_mse"]:.6f}',
        "LearnedMAE": f'{result["hardware_learned_mae"]:.5f}',
        "MSEGain": f'{100 * result["hardware_relative_mse_reduction"]:.1f}',
        "MSEGainLow": f'{100 * result["hardware_relative_mse_reduction_ci_low"]:.1f}',
        "MSEGainHigh": f'{100 * result["hardware_relative_mse_reduction_ci_high"]:.1f}',
        "PairedWinFraction": f'{100 * result["hardware_paired_win_fraction"]:.0f}',
        "MedianFoldGain": f'{100 * result["hardware_median_fold_reduction"]:.1f}',
        "MinimumFoldGain": f'{100 * result["hardware_min_fold_reduction"]:.1f}',
        "DMRCLAlpha": f'{result["hardware_selected_alpha"]:.3f}',
        "DMRCLRankTerm": f'{result["hardware_rank_counterfactual_term"]:.4f}',
        "DMRCLDepthTerm": f'{result["hardware_depth_monotonicity_term"]:.4f}',
        "ToggleMSE": f'{result["hardware_toggle_mse"]:.6f}',
        "ToggleMAE": f'{result["hardware_toggle_mae"]:.5f}',
        "ToggleGain": (
            f'{100 * result["hardware_toggle_over_legacy_relative_mse_reduction"]:.1f}'
        ),
        "ToggleGainLow": f'{100 * result["hardware_toggle_over_legacy_ci_low"]:.1f}',
        "ToggleGainHigh": f'{100 * result["hardware_toggle_over_legacy_ci_high"]:.1f}',
        "ToggleWins": (
            f'{100 * result["hardware_toggle_over_legacy_paired_win_fraction"]:.0f}'
        ),
        "AuditedAlpha": f'{result["hardware_audited_selected_alpha"]:.3f}',
        "ToggleCompatibleGroups": str(
            int(result["hardware_toggle_compatible_groups"])
        ),
        "ToggleTotalGroups": str(int(result["hardware_toggle_total_groups"])),
        "ToggleGridGap": (
            f'{result["hardware_toggle_max_grid_objective_gap_bound"]:.3e}'
        ),
        "HMatError": (
            r"\ensuremath{<10^{-12}}"
            if result["hmat_sampled_row_relative_error"] < 1e-12
            else f'{result["hmat_sampled_row_relative_error"]:.2e}'
        ),
        "HMatStorage": f'{100 * result["hmat_storage_fraction"]:.1f}',
        "HMatWorkRatio": f'{result["hmat_matmat_work_ratio"]:.2f}',
        "HMatPeakBlock": f'{100 * result["hmat_peak_block_fraction"]:.4f}',
        "HMatQueryFraction": f'{100 * result["hmat_entry_query_fraction"]:.2f}',
        "ButterflyRatio": f'{result["butterfly_work_ratio"]:.1f}',
        "StressRows": f'{int(result["stress_unique_weighted_rows"]):,}',
        "StressShots": f'{int(result["stress_records"]):,}',
        "StressBaselineLoss": f'{result["stress_baseline_loss"]:.4f}',
        "StressLogisticLoss": f'{result["stress_logistic_only_loss"]:.4f}',
        "StressLearnedLoss": f'{result["stress_learned_loss"]:.4f}',
        "StressHMatError": f'{result["stress_hmat_sampled_row_relative_error"]:.2e}',
        "StressHMatStorage": f'{100 * result["stress_hmat_storage_fraction"]:.3f}',
        "StressHMatWorkRatio": f'{result["stress_hmat_matmat_work_ratio"]:.1f}',
    }
    lines = [f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in commands.items()]
    (output / "results.tex").write_text("\n".join(lines) + "\n")


def verify(result: dict) -> None:
    expected_path = PACKAGE_ROOT / "results" / "results.json"
    if not expected_path.exists():
        raise SystemExit("No committed results.json; run with --write first")
    expected = json.loads(expected_path.read_text())
    actual = rounded(result)
    if expected.keys() != actual.keys():
        raise SystemExit("Result key mismatch")
    for key, value in actual.items():
        old = expected[key]
        if isinstance(value, float):
            if not math.isclose(value, old, rel_tol=1e-9, abs_tol=1e-9):
                raise SystemExit(f"Mismatch for {key}: expected {old}, got {value}")
        elif value != old:
            raise SystemExit(f"Mismatch for {key}: expected {old}, got {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce and verify the manuscript evidence")
    parser.add_argument("--write", action="store_true", help="write locked JSON and TeX result artifacts")
    parser.add_argument("--verify", action="store_true", help="compare a fresh run with locked results")
    args = parser.parse_args()
    result = reproduce()
    if args.write:
        write_outputs(result)
    if args.verify:
        verify(result)
    print(json.dumps(rounded(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
