import inspect
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from baseline.baseline import training_v2


ROOT = Path(__file__).resolve().parents[2]


class ChemicalFeatureComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = training_v2.load_artifact_bundle(ROOT)

    def test_four_component_modes_have_exact_numeric_and_mask_boundaries(self):
        values = np.arange(30, dtype=np.float32).reshape(3, 10) + 1
        masks = np.ones_like(values, dtype=bool)
        masks[0, 1] = False
        full, full_mask, _ = training_v2.apply_chemical_feature_components(values, masks, 6, "full")
        morgan, morgan_mask, _ = training_v2.apply_chemical_feature_components(values, masks, 6, "morgan_only")
        descriptor, descriptor_mask, _ = training_v2.apply_chemical_feature_components(values, masks, 6, "descriptor_only")
        none, none_mask, _ = training_v2.apply_chemical_feature_components(values, masks, 6, "none")
        np.testing.assert_array_equal(full, values)
        np.testing.assert_array_equal(full_mask, masks)
        np.testing.assert_array_equal(morgan[:, :6], values[:, :6])
        np.testing.assert_array_equal(morgan_mask[:, :6], masks[:, :6])
        self.assertFalse(np.any(morgan[:, 6:]))
        self.assertFalse(np.any(morgan_mask[:, 6:]))
        self.assertFalse(np.any(descriptor[:, :6]))
        self.assertFalse(np.any(descriptor_mask[:, :6]))
        np.testing.assert_array_equal(descriptor[:, 6:], values[:, 6:])
        np.testing.assert_array_equal(descriptor_mask[:, 6:], masks[:, 6:])
        self.assertFalse(np.any(none))
        self.assertFalse(np.any(none_mask))

    def test_full_equals_existing_correct_and_none_equals_existing_zero(self):
        correct = training_v2.make_chemical_feature_variant(self.artifacts, "correct", 20260814)
        legacy_zero = training_v2.make_chemical_feature_variant(self.artifacts, "zero", 20260814)
        full, full_mask, _ = training_v2.apply_chemical_feature_components(
            correct[0], correct[1], self.artifacts.morgan_dim, "full",
        )
        none, none_mask, _ = training_v2.apply_chemical_feature_components(
            correct[0], correct[1], self.artifacts.morgan_dim, "none",
        )
        np.testing.assert_array_equal(full, correct[0])
        np.testing.assert_array_equal(full_mask, correct[1])
        np.testing.assert_array_equal(none, legacy_zero[0])
        np.testing.assert_array_equal(none_mask, legacy_zero[1])

    def test_identity_quality_flags_are_unchanged_across_components(self):
        meta = training_v2.load_train_val_metadata(ROOT)
        train_ids = meta.index[meta["split_final"].eq("train")]
        vocab = training_v2.fit_category_vocabulary(meta, train_ids)
        ids = meta.index[meta["split_final"].eq("val_chem_only")][:8]
        batches = {
            mode: training_v2.build_batch(
                meta, ids, self.artifacts, vocab, chemical_mode="correct",
                chemical_feature_components=mode,
            )
            for mode in training_v2.CHEMICAL_FEATURE_COMPONENTS
        }
        reference = batches["full"]
        for batch in batches.values():
            self.assertTrue(torch.equal(batch.chemical_mapping, reference.chemical_mapping))
            self.assertTrue(torch.equal(batch.chemical_confidence, reference.chemical_confidence))
            self.assertTrue(torch.equal(batch.chemical_structure_valid, reference.chemical_structure_valid))

    def test_same_entity_is_consistent_between_train_and_validation(self):
        meta = training_v2.load_train_val_metadata(ROOT)
        train = meta.loc[meta["split_final"].eq("train")]
        validation = meta.loc[meta["split_final"].eq("val_strain_only")]
        shared = sorted(set(train["perturbation_no_concentration"]) & set(validation["perturbation_no_concentration"]))
        chemical = next(name for name in shared if name.lower() not in training_v2.CONTROL_NAMES)
        ids = [
            train.index[train["perturbation_no_concentration"].eq(chemical)][0],
            validation.index[validation["perturbation_no_concentration"].eq(chemical)][0],
        ]
        vocab = training_v2.fit_category_vocabulary(meta, train.index)
        for mode in training_v2.CHEMICAL_FEATURE_COMPONENTS:
            batch = training_v2.build_batch(
                meta, ids, self.artifacts, vocab, chemical_mode="correct",
                chemical_feature_components=mode,
            )
            self.assertTrue(torch.equal(batch.morgan[0], batch.morgan[1]))
            self.assertTrue(torch.equal(batch.descriptors[0], batch.descriptors[1]))
            self.assertTrue(torch.equal(batch.chemical_valid_mask[0], batch.chemical_valid_mask[1]))

    def test_component_interface_does_not_accept_or_read_protein_labels(self):
        signature = inspect.signature(training_v2.apply_chemical_feature_components)
        self.assertFalse(any("label" in name or "proteome" in name for name in signature.parameters))
        source = inspect.getsource(training_v2.apply_chemical_feature_components)
        self.assertNotIn("read_csv", source)
        self.assertNotIn("split_final", source)

    def test_stage_2_4_configs_keep_entity_mode_separate_from_components(self):
        expected = {
            "model_v2_huber_no_batch.yaml": "full",
            "model_v2_huber_no_batch_morgan_only.yaml": "morgan_only",
            "model_v2_huber_no_batch_descriptor_only.yaml": "descriptor_only",
        }
        for name, components in expected.items():
            config = json.loads((ROOT / "baseline/configs" / name).read_text(encoding="utf-8"))
            self.assertEqual(config["model"]["chemical_mode"], "correct")
            self.assertEqual(config["model"]["chemical_feature_components"], components)
            self.assertFalse(config["model"]["batch_enabled"])
            self.assertFalse(config["model"]["similarity_enabled"])
            self.assertEqual(config["loss"]["fc_weight"], 0)


if __name__ == "__main__":
    unittest.main()
