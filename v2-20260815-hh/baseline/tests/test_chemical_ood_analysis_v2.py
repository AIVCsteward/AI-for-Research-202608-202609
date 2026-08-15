import inspect
import unittest

import numpy as np
import pandas as pd

from baseline import chemical_ood_analysis_v2 as ood
from baseline.baseline import chemical_ood_analysis_v2 as canonical_ood


class ChemicalOodAnalysisContractTests(unittest.TestCase):
    def test_training_drugs_come_only_from_split_final_train(self):
        meta = pd.DataFrame({
            "split_final": ["train", "train", "train", "val_chem_only", "val_both"],
            "perturbation_no_concentration": ["Drug A", "Water", "Quality Control", "Drug B", "Drug C"],
        })
        self.assertEqual(ood.select_training_drugs(meta), ("Drug A",))

    def test_similarity_function_has_no_label_input(self):
        parameters = inspect.signature(ood.compute_similarity_covariates).parameters
        self.assertFalse(any("label" in name or "proteome" in name for name in parameters))
        matrix = np.array([[0.2, 0.7], [0.8, 0.1]], dtype=np.float32)
        rows = pd.DataFrame({"raw_name": ["Train A", "Train B"]})
        columns = pd.DataFrame({"raw_name": ["Val A", "Val B"]})
        result, audit = ood.compute_similarity_covariates(
            matrix, rows, columns, ["Train A"], ["Val A", "Val B"],
        )
        self.assertEqual(result["Val A"]["nearest_train_drug"], "Train A")
        self.assertAlmostEqual(result["Val B"]["max_train_morgan_tanimoto"], 0.7)
        self.assertFalse(audit["validation_label_input_accepted"])

    def test_analysis_source_names_only_train_val_proteome(self):
        source = inspect.getsource(canonical_ood)
        self.assertIn("load_label_frames", source)
        forbidden = "proteome" + "_raw_test"
        self.assertNotIn(forbidden, source)

    def test_correct_zero_alignment_requires_same_ids_masks_and_targets(self):
        base = ood.VariantEvaluation(
            ("a", "b"), np.zeros((2, 2)), np.zeros((2, 2)),
            np.ones((2, 2)), np.array([[True, False], [True, True]]),
        )
        same = ood.VariantEvaluation(
            ("a", "b"), np.ones((2, 2)), np.ones((2, 2)),
            np.ones((2, 2)), np.array([[True, False], [True, True]]),
        )
        ood.assert_variant_alignment(base, same)
        changed_mask = ood.VariantEvaluation(
            same.sample_ids, same.prediction, same.delta_response, same.target,
            np.ones((2, 2), dtype=bool),
        )
        with self.assertRaisesRegex(ValueError, "masks"):
            ood.assert_variant_alignment(base, changed_mask)

    def test_each_validation_drug_occurs_once_per_scenario(self):
        table = pd.DataFrame({
            "scenario": ["val_chem_only", "val_both"],
            "chemical_name": ["Drug A", "Drug A"],
        })
        ood.validate_unique_entity_rows(table)
        duplicate = pd.concat([table, table.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "more than once"):
            ood.validate_unique_entity_rows(duplicate)

    def test_spearman_reports_actual_finite_drug_count(self):
        result = ood.spearman_correlation([1, 2, np.nan, 3], [3, 2, 4, 1])
        self.assertEqual(result["n_drugs"], 3)
        self.assertAlmostEqual(result["spearman_rho"], -1.0)


if __name__ == "__main__":
    unittest.main()
