from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from hqml_drift_forecasting import experiment


class ReproductionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = experiment.reproduce()
        cls.actual = experiment.rounded(cls.raw)

    def test_locked_results_reproduce(self) -> None:
        actual = self.actual
        path = Path(experiment.PACKAGE_ROOT) / "results" / "results.json"
        expected = json.loads(path.read_text())
        self.assertEqual(actual.keys(), expected.keys())
        for key, value in actual.items():
            if isinstance(value, float):
                self.assertTrue(
                    math.isclose(value, expected[key], rel_tol=1e-9, abs_tol=1e-10),
                    msg=f"{key}: {value} != {expected[key]}",
                )
            else:
                self.assertEqual(value, expected[key], msg=key)

    def test_legacy_method_only_beats_the_weak_ridge_comparator(self) -> None:
        result = self.raw
        self.assertEqual(result["hardware_experiments"], 130)
        self.assertEqual(result["hardware_shots"], 6_500_000)
        self.assertEqual(result["hardware_license"], "CC BY 4.0")
        self.assertGreater(result["hardware_selected_alpha"], 0.0)
        self.assertLess(result["hardware_learned_mse"], result["hardware_baseline_mse"])
        self.assertGreater(result["hardware_min_fold_reduction"], 0.0)
        self.assertGreater(result["hardware_relative_mse_reduction_ci_low"], 0.0)
        self.assertEqual(result["hardware_paired_win_fraction"], 1.0)

    def test_standard_saturation_baseline_falsifies_the_old_headline(self) -> None:
        result = self.raw
        for toggle, legacy in zip(
            result["hardware_fold_toggle_mses"],
            result["hardware_fold_legacy_mses"],
        ):
            self.assertLess(toggle, legacy)
        self.assertEqual(result["hardware_fold_audited_alphas"], [0.0] * 4)
        self.assertLess(
            result["hardware_toggle_mse"], result["hardware_learned_mse"]
        )
        self.assertGreater(result["hardware_toggle_over_legacy_ci_low"], 0.0)
        self.assertEqual(result["hardware_toggle_compatible_groups"], 0)
        self.assertEqual(result["hardware_toggle_total_groups"], 10)

    def test_structured_products_are_verified(self) -> None:
        result = self.raw
        self.assertTrue(result["matrix_free"])
        self.assertFalse(result["global_dense_matrix_materialized"])
        self.assertLess(result["hmat_sampled_row_relative_error"], 0.05)
        self.assertLess(result["hmat_peak_block_fraction"], 0.30)
        self.assertLess(result["butterfly_inverse_error"], 1e-10)
        self.assertGreater(result["hmat_matmat_work_ratio"], 1.0)
        self.assertGreaterEqual(result["hardware_selected_alpha"], 0.0)
        self.assertLessEqual(result["hardware_selected_alpha"], 1.0)

    def test_depth_monotonicity_penalty_is_active_on_rejected_candidates(self) -> None:
        raw, target, rounds, _ = experiment._hardware_arrays()
        fold = experiment._rolling_hardware_fold(
            raw,
            target,
            rounds,
            fit_max=15,
            development_round=17,
            test_low=19,
            test_high=25,
            seed=experiment.CONFIG["seed"] + 15,
        )
        reference = fold["diagnostic_monotone_reference"]
        pure_prior = fold["diagnostic_monotone_prior_high"]
        self.assertTrue(
            any(prior < observed for prior, observed in zip(pure_prior, reference))
        )
        self.assertEqual(fold["depth_monotonicity_term"], 0.0)
        self.assertEqual(
            self.raw["hardware_depth_monotonicity_weight"],
            0.25,
        )

    def test_large_simulated_stress_case_remains_matrix_free(self) -> None:
        result = self.raw
        self.assertEqual(result["stress_unique_weighted_rows"], 75_598)
        self.assertTrue(result["stress_matrix_free"])
        self.assertFalse(result["stress_global_dense_matrix_materialized"])
        self.assertLess(result["stress_hmat_sampled_row_relative_error"], 0.01)
        self.assertLess(result["stress_hmat_storage_fraction"], 0.01)
        self.assertGreater(result["stress_hmat_matmat_work_ratio"], 100.0)

    def test_wheel_payload_contains_primary_data_and_locked_results(self) -> None:
        package = Path(experiment.SOURCE_ROOT)
        hardware = (
            package / "data" / "source" / "google_qec3v5_experiment_summary.csv"
        )
        locked = package / "results" / "results.json"
        self.assertTrue(hardware.is_file())
        self.assertTrue(locked.is_file())
        self.assertEqual(
            experiment.sha256(hardware),
            "15ffb8c7773a4fa5d1d498bbe01fd929334c0c13da69ab81704ad13beadde5f5",
        )
        self.assertEqual(json.loads(locked.read_text()).keys(), self.actual.keys())


if __name__ == "__main__":
    unittest.main()
