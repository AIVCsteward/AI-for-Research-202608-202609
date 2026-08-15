import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.baseline import drug_group_cv_v2 as cv
from baseline.baseline import training_v2


ROOT = Path(__file__).resolve().parents[2]


class DrugGroupCVContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = training_v2.load_artifact_bundle(ROOT)
        cls.meta = training_v2.load_train_val_metadata(ROOT)
        cls.eligible, cls.fold_table, cls.splits = cv.build_cv_manifest(
            cls.meta, cls.artifacts, n_folds=5, seed=20260814,
        )

    def test_same_drug_never_crosses_outer_fold_and_invalid_entities_are_excluded(self):
        eligible_rows = self.fold_table.loc[self.fold_table["cv_eligible"]]
        self.assertEqual(len(eligible_rows), 34)
        self.assertFalse(eligible_rows["outer_fold"].isna().any())
        self.assertFalse(eligible_rows["chemical_name"].duplicated().any())
        excluded = set(self.fold_table.loc[~self.fold_table["cv_eligible"], "chemical_name"])
        self.assertEqual(excluded, {"Cisplatin", "NaCl", "Tunicamycin"})
        self.assertFalse(self.fold_table.isna().any().any())
        self.assertTrue((self.fold_table.loc[~self.fold_table["cv_eligible"], "outer_fold"] == -1).all())

    def test_outer_hidden_groups_are_absent_from_training_and_inner_monitor(self):
        universe = set(self.eligible)
        for split in self.splits.values():
            train = set(split["train_drugs"])
            inner = set(split["inner_validation_drugs"])
            outer = set(split["outer_hidden_drugs"])
            self.assertFalse(train & inner)
            self.assertFalse(train & outer)
            self.assertFalse(inner & outer)
            self.assertEqual(train | inner | outer, universe)
            self.assertFalse(set(split["outer_hidden_ids"]) & set(split["train_ids"]))
            self.assertFalse(set(split["outer_hidden_ids"]) & set(split["inner_validation_ids"]))

    def test_five_folds_are_seed_stable(self):
        eligible2, table2, splits2 = cv.build_cv_manifest(
            self.meta, self.artifacts, n_folds=5, seed=20260814,
        )
        self.assertEqual(self.eligible, eligible2)
        np.testing.assert_array_equal(self.fold_table["outer_fold"], table2["outer_fold"])
        self.assertEqual(self.splits, splits2)

    def test_none_and_morgan_share_sample_order_and_stage_b_initialization(self):
        split = self.splits[0]
        self.assertEqual(
            cv.sample_order_sha256(split["train_ids"]),
            cv.sample_order_sha256(split["train_ids"]),
        )
        config = training_v2.load_config(
            ROOT / "baseline/configs/model_v2_stage2_6_drug_group_cv.yaml"
        )["stage2_6"]
        train_ids = self.meta.index[self.meta["split_final"].eq("train")]
        vocab = training_v2.fit_category_vocabulary(self.meta, train_ids)
        names = self.meta.loc[train_ids, cv.CHEMICAL_COLUMN].astype(str).str.lower()
        control_ids = train_ids[names.isin(training_v2.CONTROL_NAMES)]
        labels, masks = training_v2.load_label_frames(self.meta, self.artifacts, ROOT)
        mean, _ = training_v2.fit_control_protein_anchor(
            self.meta, labels, masks, control_ids,
        )
        checkpoint = Path(config["fixed_stage_a_checkpoint"]["path"])
        model1, _ = cv._model_from_stage_a(
            config, self.artifacts, vocab, torch.from_numpy(mean), checkpoint, "cpu",
        )
        model2, _ = cv._model_from_stage_a(
            config, self.artifacts, vocab, torch.from_numpy(mean), checkpoint, "cpu",
        )
        hash1 = training_v2.module_parameters_sha256(model1, ("chemical_encoder", "response_branch"))
        hash2 = training_v2.module_parameters_sha256(model2, ("chemical_encoder", "response_branch"))
        self.assertEqual(hash1, hash2)

    def test_tanimoto_neighbor_candidates_are_current_fold_gradient_train_only(self):
        similarity, rows, columns, _ = cv.load_tanimoto(ROOT)
        for fold, split in self.splits.items():
            nearest, audit = cv.compute_similarity_covariates(
                similarity, rows, columns,
                split["train_drugs"], split["outer_hidden_drugs"],
            )
            allowed = set(split["train_drugs"])
            forbidden = set(split["inner_validation_drugs"]) | set(split["outer_hidden_drugs"])
            self.assertEqual(audit["training_drug_count"], len(allowed))
            self.assertFalse(audit["validation_label_input_accepted"])
            for result in nearest.values():
                self.assertIn(result["nearest_train_drug"], allowed)
                self.assertNotIn(result["nearest_train_drug"], forbidden)

    def test_outer_labels_are_sliced_only_after_best_checkpoint_reload(self):
        source = inspect.getsource(cv.evaluate_outer_after_best_restore)
        self.assertLess(source.index("load_checkpoint"), source.index("_loader_for"))
        self.assertLess(source.index("model.load_state_dict"), source.index("_loader_for"))

    def test_cv_sources_do_not_name_test_proteome_or_official_validation_scenarios(self):
        source = inspect.getsource(cv)
        self.assertNotIn("proteome" + "_raw_test", source)
        for scenario in training_v2.VAL_SCENARIOS:
            self.assertNotIn(f'"{scenario}"', source)

    def test_config_freezes_authorized_comparison(self):
        config = training_v2.load_config(
            ROOT / "baseline/configs/model_v2_stage2_6_drug_group_cv.yaml"
        )["stage2_6"]
        self.assertEqual(config["component_modes"], ["none", "morgan_only"])
        self.assertEqual(config["loss"]["fc_weight"], 0.0)
        self.assertFalse(config["model"]["descriptor_enabled"])
        self.assertFalse(config["model"]["batch_enabled"])
        self.assertFalse(config["model"]["similarity_enabled"])
        self.assertFalse(config["official_validation_for_checkpoint_selection"])

    def test_formal_outputs_are_finite_aligned_and_fair(self):
        per_entity = pd.read_csv(
            ROOT / "reports/model_v2_stage2/stage2_6_drug_cv_per_entity.csv"
        )
        self.assertEqual(len(per_entity), 68)
        self.assertEqual(
            len(per_entity.drop_duplicates(["outer_fold", "chemical_name"])), 34,
        )
        self.assertFalse(per_entity.duplicated(["outer_fold", "chemical_name", "model"]).any())
        numeric = [
            "n_samples", "rmse", "mae", "response_target_cosine",
            "response_target_pearson", "alpha_star",
            "max_train_morgan_tanimoto", "rmse_gain",
        ]
        self.assertTrue(np.isfinite(per_entity[numeric].to_numpy(np.float64)).all())
        training_root = ROOT / "reports/model_v2_stage2/formal_stage2_6_drug_cv_seed_20260814"
        for fold in range(5):
            summaries = []
            for mode in cv.COMPONENT_MODES:
                summary = training_v2.read_json(
                    training_root / f"fold_{fold}" / mode / "training_summary.json"
                )
                summaries.append(summary)
                self.assertFalse(summary["outer_labels_used_for_training_or_early_stopping"])
                self.assertFalse(summary["official_validation_used"])
                self.assertFalse(summary["test_proteome_opened"])
                self.assertEqual(
                    summary["outer_evaluation_sequence"][-1],
                    "outer_labels_sliced_after_restore",
                )
            self.assertEqual(
                summaries[0]["stage_b_initial_chemical_response_sha256"],
                summaries[1]["stage_b_initial_chemical_response_sha256"],
            )
            self.assertEqual(
                summaries[0]["train_sample_order_sha256"],
                summaries[1]["train_sample_order_sha256"],
            )

    def test_few_drug_concentration_rule_is_explicit(self):
        table = pd.DataFrame({
            "outer_fold": range(7),
            "chemical_name": [f"d{i}" for i in range(7)],
            "model": ["none"] * 7,
            "rmse_gain": [-1.0, -0.8, -0.6, 0.1, 0.2, 0.3, 0.4],
        })
        stats = cv._gain_decision_statistics(table)
        self.assertTrue(stats["few_drugs_dominate"])
        self.assertGreater(stats["median_gain"], 0)
        self.assertGreater(stats["mean_gain_excluding_three_most_harmful"], 0)


if __name__ == "__main__":
    unittest.main()
