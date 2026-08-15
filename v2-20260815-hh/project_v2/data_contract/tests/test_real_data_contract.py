from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))
import audit_data_contract as audit  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[3] / "WAYB_WAYC"


class RealDataContractTests(unittest.TestCase):
    def test_real_files_freeze_expected_counts_and_no_test_label_access(self):
        result, feature_contract = audit.audit(audit.default_paths(DATA_DIR))
        self.assertEqual(
            result["split_counts"],
            {
                "test_both": 1129,
                "test_chem_only": 1640,
                "test_strain_only": 1534,
                "test_time": 151,
                "train": 5920,
                "val_both": 269,
                "val_chem_only": 1065,
                "val_strain_only": 1547,
                "val_time": 157,
            },
        )
        self.assertEqual(feature_contract["n_source_proteins"], 5243)
        self.assertEqual(feature_contract["n_proteins"], 4422)
        self.assertTrue(result["test_label_access"].startswith("sample_ID column only"))
        self.assertFalse(feature_contract["leakage_guard"]["validation_labels_used"])
        self.assertFalse(feature_contract["leakage_guard"]["test_labels_used"])


if __name__ == "__main__":
    unittest.main()
