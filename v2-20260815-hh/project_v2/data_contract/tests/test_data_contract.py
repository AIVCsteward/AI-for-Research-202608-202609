from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))
import audit_data_contract as audit  # noqa: E402


def base_row(sample_id: str, split: str, treatment: str, pert_id: str = "#3"):
    return {
        "sample_ID": sample_id,
        "split_final": split,
        "perturbation_no_concentration": treatment,
        "pert_id": pert_id,
        "data_source": "S",
        "Strains": "strain",
        "Medium": "medium",
        "Temperature": 30,
        "pert_time": 15,
        "pert_time_unit": "min",
        "instrument": "I",
        "Yeast_cell_plate": "P",
    }


class DataContractTests(unittest.TestCase):
    def test_aligns_by_sample_id_not_source_row_order(self):
        meta = pd.DataFrame(
            {"sample_ID": ["b", "a"], "split_final": ["train", "train"]}
        )
        prot = pd.DataFrame({"sample_ID": ["a", "b"], "P": [1.0, 2.0]})
        aligned_meta, aligned_prot = audit.align_by_sample_id(meta, prot)
        self.assertEqual(aligned_meta.index.tolist(), ["b", "a"])
        self.assertEqual(aligned_prot.index.tolist(), ["b", "a"])
        self.assertEqual(aligned_prot["P"].tolist(), [2.0, 1.0])

    def test_label_statistics_reject_validation_ids(self):
        meta = pd.DataFrame(
            {"sample_ID": ["train", "val"], "split_final": ["train", "val_both"]}
        )
        for bad_ids in (["val"], ["train", "val"]):
            with self.subTest(bad_ids=bad_ids):
                with self.assertRaisesRegex(audit.ContractError, "split_final=train only"):
                    audit.require_train_fit_ids(meta, bad_ids)

    def test_missing_threshold_is_strict_and_uses_only_train_ids(self):
        meta = pd.DataFrame(
            {
                "sample_ID": ["t1", "t2", "t3", "t4", "t5", "v"],
                "split_final": ["train"] * 5 + ["val_both"],
            }
        )
        prot = pd.DataFrame(
            {
                "sample_ID": meta["sample_ID"],
                "boundary": [1.0, np.nan, np.nan, np.nan, np.nan, 9.0],
                "retained": [1.0, 2.0, np.nan, np.nan, np.nan, np.nan],
            }
        )
        kept, rates = audit.select_proteins(
            meta, prot, ["t1", "t2", "t3", "t4", "t5"]
        )
        self.assertAlmostEqual(rates["boundary"], 0.8)
        self.assertNotIn("boundary", kept)
        self.assertEqual(kept, ["retained"])

    def test_zero_negative_and_infinite_raw_values_are_invalid(self):
        values = np.array([[1.0, 0.0, -1.0, np.inf, -np.inf, np.nan]])
        transformed, mask = audit.log2_with_mask(values)
        self.assertEqual(mask.tolist(), [[True, False, False, False, False, False]])
        self.assertEqual(transformed[0, 0], 0.0)
        self.assertTrue(np.isnan(transformed[0, 1:]).all())

    def test_official_control_matching_fails_closed_without_mapping(self):
        meta = pd.DataFrame(
            [
                base_row("t", "train", "drug", "#3"),
                base_row("w", "train", "Water", "#1"),
                base_row("d", "train", "DMSO", "#2"),
            ]
        )
        pairs, failures = audit.build_matched_control_pairs(
            meta, ["t"], ["w", "d"], control_mapping={}
        )
        self.assertTrue(pairs.empty)
        self.assertEqual(
            failures.to_dict(orient="records"),
            [
                {
                    "treatment_sample_ID": "t",
                    "reason": "unconfirmed_control_mapping",
                    "mapping_key": "#3",
                }
            ],
        )

    def test_matching_uses_expected_solvent_and_every_exact_key(self):
        meta = pd.DataFrame(
            [
                base_row("t", "train", "drug", "#3"),
                base_row("w", "train", "Water", "#1"),
                base_row("d", "train", "DMSO", "#2"),
            ]
        )
        pairs, failures = audit.build_matched_control_pairs(
            meta, ["t"], ["w", "d"], control_mapping={"#3": "Water"}
        )
        self.assertTrue(failures.empty)
        self.assertEqual(pairs.iloc[0]["control_sample_ids"], ("w",))
        changed = meta.copy()
        changed.loc[changed.sample_ID.eq("w"), "pert_time_unit"] = "hour"
        pairs, failures = audit.build_matched_control_pairs(
            changed, ["t"], ["w", "d"], control_mapping={"#3": "Water"}
        )
        self.assertTrue(pairs.empty)
        self.assertEqual(failures.iloc[0]["reason"], "no_exact_expected_control")

    def test_fc_and_metrics_use_treatment_control_common_mask(self):
        result = audit.compute_fc_arrays(
            np.array([[4.0, 99.0, 8.0, 12.0]]),
            np.array([[3.0, -999.0, 7.0, np.nan]]),
            np.array([[1.0, -88.0, 2.0, 3.0]]),
            np.array([[True, False, True, True]]),
            np.array([[True, True, False, True]]),
        )
        self.assertEqual(result["fc_mask"].tolist(), [[True, False, False, False]])
        self.assertEqual(result["fc_true"][0, 0], 3.0)
        self.assertEqual(result["fc_pred_official"][0, 0], 2.0)
        self.assertTrue(np.isnan(result["fc_true"][0, 1:]).all())

    def test_control_replicates_are_averaged_protein_wise_over_observed_values(self):
        values = np.array([[1.0, 20.0, np.nan], [3.0, 999.0, np.nan]])
        mask = np.array([[True, True, False], [True, False, False]])
        mean, observed = audit.aggregate_observed_controls(values, mask)
        np.testing.assert_allclose(mean[:2], [2.0, 20.0])
        self.assertTrue(np.isnan(mean[2]))
        self.assertEqual(observed.tolist(), [True, True, False])

    def test_masked_metrics_ignore_fill_values_and_reject_shape_errors(self):
        mask = np.array([True, True, False, False])
        truth_a = np.array([1.0, 3.0, 1e9, -1e9])
        pred_a = np.array([1.0, 2.0, -8e8, 7e8])
        truth_b = np.array([1.0, 3.0, -123.0, 456.0])
        pred_b = np.array([1.0, 2.0, 789.0, -321.0])
        self.assertEqual(
            audit.masked_r2(truth_a, pred_a, mask),
            audit.masked_r2(truth_b, pred_b, mask),
        )
        self.assertEqual(
            audit.masked_pearson(truth_a, pred_a, mask),
            audit.masked_pearson(truth_b, pred_b, mask),
        )
        with self.assertRaisesRegex(audit.ContractError, "identical shapes"):
            audit.masked_r2(np.ones(2), np.ones(3), np.ones(2, dtype=bool))

    def test_test_proteome_reader_requests_sample_id_only(self):
        observed = {}

        def fake_read_csv(path, **kwargs):
            observed.update(kwargs)
            return pd.DataFrame({"sample_ID": ["x"]})

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(audit.pd, "read_csv", side_effect=fake_read_csv):
                result = audit.read_proteome_sample_ids(Path(directory) / "test.csv")
        self.assertEqual(observed, {"usecols": ["sample_ID"]})
        self.assertEqual(result.columns.tolist(), ["sample_ID"])

    def test_fit_proteome_reader_skips_validation_cells_before_parsing(self):
        metadata = pd.DataFrame(
            {"sample_ID": ["t", "v"], "split_final": ["train", "val_both"]}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.csv"
            path.write_text(
                "sample_ID,P\nt,2\nv,NOT_A_NUMBER\n", encoding="utf-8"
            )
            result = audit.read_fit_proteome_rows(path, metadata, ["t"])
        self.assertEqual(result["sample_ID"].tolist(), ["t"])
        self.assertEqual(result["P"].tolist(), [2])

    def test_audit_does_not_hash_test_proteome(self):
        source = MODULE_DIR.joinpath("audit_data_contract.py").read_text(encoding="utf-8")
        self.assertIn('if field == "proteome_test"', source)
        self.assertIn("deliberately_not_computed_to_avoid_reading_test_truth_bytes", source)

    def test_masked_mean_rejects_validation_even_when_values_are_available(self):
        meta = pd.DataFrame(
            {"sample_ID": ["t", "v"], "split_final": ["train", "val_chem_only"]}
        )
        values = pd.DataFrame({"P": [1.0, 1000.0]}, index=["t", "v"])
        self.assertEqual(audit.fit_masked_mean(meta, values, ["t"])["P"], 1.0)
        with self.assertRaises(audit.ContractError):
            audit.fit_masked_mean(meta, values, ["t", "v"])


if __name__ == "__main__":
    unittest.main()
