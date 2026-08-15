import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.baseline import chemical_component_analysis_v2 as analysis
from baseline.baseline.normalized_response_analysis_v2 import ResponseEvaluation


class ChemicalComponentAnalysisTests(unittest.TestCase):
    def test_legacy_checkpoint_component_resolution(self):
        self.assertEqual(
            analysis.resolve_checkpoint_components({"config": {"model": {"chemical_mode": "correct"}}}),
            "full",
        )
        self.assertEqual(
            analysis.resolve_checkpoint_components({"config": {"model": {"chemical_mode": "zero"}}}),
            "none",
        )
        self.assertEqual(
            analysis.resolve_checkpoint_components({"config": {"model": {
                "chemical_mode": "correct", "chemical_feature_components": "morgan_only",
            }}}),
            "morgan_only",
        )

    def test_alignment_requires_identical_baseline_and_masks(self):
        def item(baseline, mask):
            return ResponseEvaluation(
                ("a",), np.zeros((1, 2)), np.asarray(baseline, dtype=float),
                np.zeros((1, 2)), np.ones((1, 2)), np.asarray(mask, dtype=bool),
            )
        aligned = {mode: item([[1, 2]], [[True, True]]) for mode in analysis.COMPONENT_MODES}
        analysis.assert_all_mode_alignment(aligned)
        aligned["none"] = item([[1, 3]], [[True, True]])
        with self.assertRaisesRegex(ValueError, "baseline"):
            analysis.assert_all_mode_alignment(aligned)

    def test_two_scenarios_are_equally_averaged_not_sample_weighted(self):
        rows = []
        for mode in analysis.COMPONENT_MODES:
            for drug_index in range(6):
                for scenario, samples, value in (
                    ("val_chem_only", 1000, 2.0), ("val_both", 1, 10.0),
                ):
                    row = {
                        "mode": mode, "chemical_name": f"drug_{drug_index}",
                        "scenario": scenario, "n_samples": samples,
                    }
                    for column in (
                        "rmse", "sample_response_rms_mean", "sample_response_rms_median",
                        "predicted_response_rms", "target_residual_rms", "response_target_norm_ratio",
                        "response_target_cosine", "response_target_pearson", "alpha_star",
                    ):
                        row[column] = value
                    rows.append(row)
        combined = analysis.combine_entity_directions_equal_scenarios(pd.DataFrame(rows))
        self.assertEqual(len(combined), 24)
        self.assertTrue(np.allclose(combined["sample_response_rms_mean"], 6.0))
        self.assertFalse(np.isclose(6.0, (1000 * 2 + 10) / 1001))

    def test_stage_a_hash_fallback_is_deterministic(self):
        state = {
            "chemical_encoder.weight": torch.tensor([[1.0, 2.0]]),
            "response_branch.bias": torch.tensor([3.0]),
            "baseline_branch.weight": torch.tensor([99.0]),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stage_a.pt"
            torch.save({"model_state": state}, path)
            first = analysis.chemical_response_state_sha256(path)
            second = analysis.chemical_response_state_sha256(path)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
