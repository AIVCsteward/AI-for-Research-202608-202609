import inspect
import unittest

import numpy as np
import pandas as pd

from baseline.baseline import normalized_response_analysis_v2 as audit


class NormalizedResponseAuditTests(unittest.TestCase):
    def test_rms_denominator_is_common_valid_position_count(self):
        response = np.array([[3.0, 4.0, 100.0], [0.0, 12.0, 5.0]])
        mask = np.array([[True, True, False], [False, True, True]])
        rms, l2, counts = audit.per_sample_response_amplitudes(response, mask)
        np.testing.assert_allclose(rms, [np.sqrt(25 / 2), np.sqrt(169 / 2)])
        np.testing.assert_allclose(l2, [5.0, 13.0])
        np.testing.assert_array_equal(counts, [2, 2])

    def test_replication_does_not_change_normalized_rms(self):
        response = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 2.0]])
        mask = np.ones_like(response, dtype=bool)
        original = audit.summarize_response_amplitude(response, mask)
        replicated = audit.summarize_response_amplitude(
            np.tile(response, (5, 1)), np.tile(mask, (5, 1)),
        )
        self.assertAlmostEqual(
            original["sample_response_rms_mean"], replicated["sample_response_rms_mean"],
        )
        self.assertAlmostEqual(
            original["sample_response_rms_median"], replicated["sample_response_rms_median"],
        )

    def test_raw_l2_scales_as_square_root_of_replication(self):
        response = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.ones_like(response, dtype=bool)
        original = audit.summarize_response_amplitude(response, mask)
        replicated = audit.summarize_response_amplitude(
            np.tile(response, (9, 1)), np.tile(mask, (9, 1)),
        )
        self.assertAlmostEqual(
            replicated["raw_cumulative_response_l2"],
            3 * original["raw_cumulative_response_l2"],
        )

    def test_six_drug_combination_is_equal_weight_not_sample_weighted(self):
        rows = []
        for drug_index in range(6):
            for scenario, n_samples, value in (
                ("val_chem_only", 1000, 2.0 + drug_index),
                ("val_both", 1, 10.0 + drug_index),
            ):
                row = {
                    "chemical_name": f"drug_{drug_index}", "scenario": scenario,
                    "n_samples": n_samples, "n_common_valid_positions": n_samples,
                    "correct_raw_cumulative_response_l2": np.sqrt(n_samples) * value,
                    "zero_raw_cumulative_response_l2": np.sqrt(n_samples),
                }
                for column in audit.MEAN_COMBINED_COLUMNS:
                    row[column] = value
                rows.append(row)
        combined = audit.combine_unique_drugs_equal_weight(pd.DataFrame(rows))
        self.assertEqual(len(combined), 6)
        first = combined.set_index("chemical_name").loc["drug_0"]
        self.assertEqual(first["correct_sample_response_rms_mean"], 6.0)
        self.assertNotAlmostEqual(first["correct_sample_response_rms_mean"], (1000 * 2 + 10) / 1001)
        self.assertTrue((combined["aggregation"] == "equal_mean_of_two_scenario_level_metrics_per_unique_drug").all())

    def test_direction_alpha_star_recovers_known_scale(self):
        response = np.array([[1.0, 2.0, 3.0]])
        baseline = np.array([[10.0, 10.0, 10.0]])
        target = baseline + 0.25 * response
        result = audit.response_direction_diagnostics(
            response, target, baseline, np.ones_like(response, dtype=bool),
        )
        self.assertAlmostEqual(result["alpha_star"], 0.25)
        self.assertAlmostEqual(result["response_target_cosine"], 1.0)
        self.assertAlmostEqual(result["response_target_pearson"], 1.0)

    def test_source_has_no_test_proteome_path_or_training_calls(self):
        source = inspect.getsource(audit)
        forbidden = "proteome" + "_raw_test"
        self.assertNotIn(forbidden, source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer.step(", source)


if __name__ == "__main__":
    unittest.main()
