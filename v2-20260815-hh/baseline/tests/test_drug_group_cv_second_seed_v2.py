import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.baseline import drug_group_cv_second_seed_v2 as comparison
from baseline.baseline import drug_group_cv_v2 as cv
from baseline.baseline import training_v2


ROOT = Path(__file__).resolve().parents[2]
FOLDS = ROOT / "reports/model_v2_stage2/stage2_6_drug_cv_folds.csv"
REFERENCE_ROOT = ROOT / "reports/model_v2_stage2/formal_stage2_6_drug_cv_seed_20260814"
EXPECTED_FOLDS_SHA = "2110ef8e8bec22cfc09237ad0e483ad30240464af4626641bb22aeb6f376ff1f"


class SecondSeedReuseContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = training_v2.load_artifact_bundle(ROOT)
        cls.meta = training_v2.load_train_val_metadata(ROOT)

    def test_frozen_folds_csv_is_directly_reused_with_exact_hash(self):
        self.assertEqual(training_v2.sha256_file(FOLDS), EXPECTED_FOLDS_SHA)
        eligible, table, splits, audit = cv.load_reused_cv_manifest(
            self.meta, self.artifacts, FOLDS, REFERENCE_ROOT,
            EXPECTED_FOLDS_SHA, n_folds=5,
        )
        self.assertEqual(len(eligible), 34)
        self.assertEqual(len(table), 37)
        self.assertEqual(set(splits), set(range(5)))
        self.assertFalse(audit["folds_regenerated"])
        self.assertTrue(audit["inner_train_ids_reused"])

    def test_reused_outer_inner_train_ids_match_seed1_audits(self):
        _, _, splits, _ = cv.load_reused_cv_manifest(
            self.meta, self.artifacts, FOLDS, REFERENCE_ROOT,
            EXPECTED_FOLDS_SHA, n_folds=5,
        )
        for fold, split in splits.items():
            audit = training_v2.read_json(
                REFERENCE_ROOT / f"fold_{fold}/none/fold_split_audit.json"
            )
            self.assertEqual(cv.sample_order_sha256(split["train_ids"]), audit["train_sample_order_sha256"])
            self.assertEqual(cv.sample_order_sha256(split["inner_validation_ids"]), audit["inner_sample_order_sha256"])
            self.assertEqual(cv.sample_order_sha256(split["outer_hidden_ids"]), audit["outer_sample_order_sha256"])

    def test_second_seed_changes_only_stochastic_seed_and_output_metadata(self):
        first = training_v2.load_config(
            ROOT / "baseline/configs/model_v2_stage2_6_drug_group_cv.yaml"
        )["stage2_6"]
        second = training_v2.load_config(
            ROOT / "baseline/configs/model_v2_stage2_6b_second_seed.yaml"
        )["stage2_6"]
        self.assertEqual(first["seed"], 20260814)
        self.assertEqual(second["seed"], 20260815)
        for field in (
            "n_outer_folds", "component_modes", "fixed_stage_a_checkpoint", "model",
            "loss", "batch_size", "weight_decay", "device", "num_workers", "stage_b",
        ):
            if field == "stage_b":
                left, right = dict(first[field]), dict(second[field])
                left.pop("name"); right.pop("name")
                self.assertEqual(left, right)
            else:
                self.assertEqual(first[field], second[field])
        self.assertEqual(second["stage_a_checkpoint_seed"], 20260814)
        self.assertTrue(second["reuse_frozen_stage2_6_splits"]["regeneration_forbidden"])

    def test_analysis_alignment_is_drug_equal_and_finite(self):
        seed1 = pd.read_csv(ROOT / "reports/model_v2_stage2/stage2_6_drug_cv_per_entity.csv")
        aligned = comparison.align_two_seed_results(seed1, seed1.copy())
        self.assertEqual(len(aligned), 34)
        self.assertTrue(aligned["gain_sign_consistent"].all())
        self.assertTrue(np.isfinite(aligned.select_dtypes(include=[np.number])).all().all())
        np.testing.assert_allclose(aligned["gain_std_across_seeds"], 0.0)

    def test_second_seed_sources_do_not_name_test_proteome(self):
        forbidden = "proteome" + "_raw_test"
        self.assertNotIn(forbidden, inspect.getsource(cv))
        self.assertNotIn(forbidden, inspect.getsource(comparison))

    def test_formal_second_seed_outputs_preserve_frozen_splits_and_are_finite(self):
        root = ROOT / "reports/model_v2_stage2/formal_stage2_6b_second_seed_20260815"
        per_entity = pd.read_csv(root / "seed_20260815_per_entity.csv")
        self.assertEqual(len(per_entity), 68)
        self.assertFalse(per_entity.isna().any().any())
        self.assertTrue(np.isfinite(per_entity.select_dtypes(include=[np.number])).all().all())
        summary = training_v2.read_json(root / "seed_20260815_summary.json")
        self.assertFalse(summary["fold_reuse_audit"]["folds_regenerated"])
        self.assertTrue(summary["fold_reuse_audit"]["inner_train_ids_reused"])
        self.assertEqual(summary["fold_reuse_audit"]["folds_sha256"], EXPECTED_FOLDS_SHA)
        self.assertFalse(summary["test_proteome_opened"])
        for fold in range(5):
            for mode in cv.COMPONENT_MODES:
                run = root / f"fold_{fold}" / mode
                self.assertTrue((run / "stage_b_best.pt").is_file())
                self.assertTrue((run / "training_history.json").is_file())
                audit2 = training_v2.read_json(run / "fold_split_audit.json")
                audit1 = training_v2.read_json(
                    REFERENCE_ROOT / f"fold_{fold}" / mode / "fold_split_audit.json"
                )
                for field in (
                    "train_sample_order_sha256", "inner_sample_order_sha256",
                    "outer_sample_order_sha256",
                ):
                    self.assertEqual(audit1[field], audit2[field])

    def test_two_seed_delivery_has_one_finite_row_per_drug(self):
        table = pd.read_csv(
            ROOT / "reports/model_v2_stage2/stage2_6b_two_seed_per_entity.csv"
        )
        self.assertEqual(len(table), 34)
        self.assertFalse(table["chemical_name"].duplicated().any())
        self.assertFalse(table.isna().any().any())
        self.assertTrue(np.isfinite(table.select_dtypes(include=[np.number])).all().all())
        summary = training_v2.read_json(
            ROOT / "reports/model_v2_stage2/stage2_6b_two_seed_summary.json"
        )
        self.assertEqual(summary["n_drugs"], 34)
        self.assertTrue(summary["fairness"]["same_outer_inner_train_sample_ids"])
        self.assertFalse(summary["test_proteome_opened"])


if __name__ == "__main__":
    unittest.main()
