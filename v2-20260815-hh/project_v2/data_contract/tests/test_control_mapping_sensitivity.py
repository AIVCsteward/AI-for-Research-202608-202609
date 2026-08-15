from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))
import audit_data_contract as contract  # noqa: E402
import audit_control_mapping_sensitivity as sensitivity  # noqa: E402


def metadata_row(sample_id: str, split: str, name: str, pert_id: str) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_ID": sample_id,
        "split_final": split,
        "perturbation_no_concentration": name,
        "pert_id": pert_id,
    }
    row.update(
        {
            "data_source": "WAYB",
            "Strains": "S1",
            "Medium": "M1",
            "Temperature": 30,
            "pert_time": 1,
            "pert_time_unit": "hour",
            "instrument": "I1",
            "Yeast_cell_plate": "P1",
        }
    )
    return row


class ControlMappingSensitivityTests(unittest.TestCase):
    def test_frozen_mapping_boundaries_and_qc(self):
        mapping = sensitivity.inferred_parity_mapping()
        self.assertEqual(len(mapping), 47)
        self.assertEqual(mapping["#1"], "Water")
        self.assertEqual(mapping["#2"], "DMSO")
        self.assertEqual(mapping["#47"], "Water")
        self.assertNotIn("#48", mapping)
        self.assertEqual(sensitivity.classify_pert_id("#48"), "Quality Control")
        with self.assertRaises(contract.ContractError):
            sensitivity.classify_pert_id("#49")

    def test_train_only_loader_skips_unparseable_validation_protein_cell(self):
        metadata = pd.DataFrame(
            [
                metadata_row("train-1", "train", "Drug A", "#3"),
                metadata_row("val-secret", "val_time", "Drug B", "#4"),
                metadata_row("train-2", "train", "Water", "#1"),
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mixed.csv"
            path.write_text(
                "sample_ID,p1\ntrain-1,4\nval-secret,NOT_A_NUMBER\ntrain-2,2\n",
                encoding="utf-8",
            )
            loaded = sensitivity.read_train_proteome_without_parsing_validation_labels(
                path, metadata, ["p1"]
            )
        self.assertEqual(loaded.index.tolist(), ["train-1", "train-2"])
        np.testing.assert_allclose(loaded["p1"].to_numpy(), [4.0, 2.0])

    def test_synthetic_fc_comparison_is_mask_aware(self):
        metadata = pd.DataFrame(
            [
                metadata_row("water", "train", "Water", "#1"),
                metadata_row("dmso", "train", "DMSO", "#2"),
                metadata_row("odd", "train", "Drug Odd", "#3"),
                metadata_row("even", "train", "Drug Even", "#4"),
                metadata_row("qc", "train", "Quality Control", "#48"),
                metadata_row("val", "val_time", "Drug Val", "#5"),
            ]
        )
        train_values = pd.DataFrame(
            {
                "sample_ID": ["water", "dmso", "odd", "even", "qc"],
                "p1": [1.0, 4.0, 4.0, 8.0, 2.0],
                "p2": [4.0, 1.0, 8.0, 4.0, 2.0],
            }
        ).set_index("sample_ID", drop=False)
        result = sensitivity.run_sensitivity(
            metadata,
            train_values,
            {"proteins": ["p1", "p2"], "generation_sha256": "synthetic"},
        )
        self.assertEqual(result["coverage"]["inferred_parity"]["matched"], 2)
        self.assertEqual(result["coverage"]["pooled_context"]["matched"], 2)
        comparison = result["comparison_on_common_matched_and_valid_mask"]
        self.assertAlmostEqual(comparison["fc_pearson_global_flattened"], -1.0)
        self.assertAlmostEqual(comparison["direction_consistency_rate"], 1.0)
        self.assertEqual(
            comparison["high_effect_on_common_subset"]["inferred_parity"][
                "sample_protein_entries"
            ],
            2,
        )
        self.assertFalse(
            result["decision_policy"]["validation_or_test_score_used_to_select_rule"]
        )
        self.assertFalse(result["data_scope"]["validation_protein_labels_read_or_used"])
        self.assertFalse(result["data_scope"]["test_proteome_opened"])

    def test_run_rejects_any_non_train_protein_rows(self):
        metadata = pd.DataFrame(
            [
                metadata_row("train", "train", "Drug A", "#3"),
                metadata_row("val", "val_time", "Drug B", "#4"),
            ]
        )
        values = pd.DataFrame(
            {"sample_ID": ["train", "val"], "p1": [1.0, 99.0]}
        ).set_index("sample_ID", drop=False)
        with self.assertRaisesRegex(contract.ContractError, "exactly split_final=train"):
            sensitivity.run_sensitivity(metadata, values, {"proteins": ["p1"]})

    def test_official_spec_remains_blocked_and_inferred_rule_is_separate(self):
        spec = json.loads(
            (MODULE_DIR / "control_matching_spec.json").read_text(encoding="utf-8")
        )
        self.assertEqual(spec["status"], "blocked_pending_organizer_confirmation")
        self.assertEqual(spec["confirmed_control_mapping"], {})
        inferred = spec["inferred_mapping_for_experimental_fc_loss"]
        self.assertEqual(inferred["status"], "inferred_mapping")
        self.assertFalse(inferred["official"])
        self.assertEqual(inferred["mapping"], sensitivity.inferred_parity_mapping())
        self.assertEqual(inferred["quality_control"], {"#48": "Quality Control"})


class RealSensitivityResultTests(unittest.TestCase):
    def test_machine_result_has_train_only_scope_and_expected_coverage(self):
        result_path = MODULE_DIR / "control_mapping_sensitivity_results.json"
        if not result_path.exists():
            self.skipTest("generate control_mapping_sensitivity_results.json first")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(result["official_mapping_status"], "blocked_pending_organizer_confirmation")
        self.assertEqual(result["data_scope"]["fit_sample_count"], 5920)
        self.assertFalse(result["data_scope"]["validation_protein_labels_read_or_used"])
        self.assertFalse(result["data_scope"]["test_proteome_opened"])
        self.assertEqual(
            result["data_scope"]["non_train_protein_cell_handling"],
            "non-train lines skipped before protein-cell tokenization; never materialized or used",
        )
        self.assertEqual(result["coverage"]["inferred_parity"]["matched"], 4785)
        self.assertEqual(result["coverage"]["pooled_context"]["matched"], 5066)
        self.assertFalse(
            result["decision_policy"]["validation_or_test_score_used_to_select_rule"]
        )


if __name__ == "__main__":
    unittest.main()
