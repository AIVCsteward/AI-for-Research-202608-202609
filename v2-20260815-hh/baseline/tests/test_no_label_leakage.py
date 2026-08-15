from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from baseline.baseline import training_v2
from baseline.baseline.losses_v2 import LossWeights
from baseline.baseline.model_v2 import V2Batch


class NoLabelLeakageTests(unittest.TestCase):
    def setUp(self):
        self.meta = pd.DataFrame({
            "sample_ID": ["t1", "t2", "v1"],
            "split_final": ["train", "train", "val_both"],
            "perturbation_no_concentration": ["DrugA", "DrugB", "DrugA"],
        }).set_index("sample_ID", drop=False)
        self.labels = pd.DataFrame([[1., 2.], [3., 4.], [100., 200.]], index=self.meta.index)
        self.masks = pd.DataFrame(True, index=self.meta.index, columns=[0, 1])

    def test_training_source_has_no_test_proteome_path(self):
        source = inspect.getsource(training_v2).lower()
        self.assertNotIn("proteome_raw_test", source)
        self.assertNotIn("*proteome", source)

    def test_label_loader_runtime_opens_train_val_only(self):
        class Contract:
            proteins = ("P1", "P2")
        class Artifacts:
            feature_contract = Contract()
        observed = []
        def guarded_read_csv(path, **kwargs):
            observed.append(str(path).lower())
            self.assertNotIn("proteome_raw_test", str(path).lower())
            return pd.DataFrame({"sample_ID": ["t1", "t2", "v1"], "P1": [2., 4., 8.], "P2": [4., 8., 16.]})
        with patch.object(training_v2.pd, "read_csv", side_effect=guarded_read_csv):
            labels, masks = training_v2.load_label_frames(self.meta, Artifacts())
        self.assertEqual(len(observed), 1)
        self.assertIn("proteome_raw_train_val", observed[0])
        self.assertEqual(labels.shape, masks.shape)

    def test_validation_ids_rejected_by_all_label_fitters(self):
        with self.assertRaises(ValueError):
            training_v2.fit_protein_mean(self.meta, self.labels, self.masks, ["t1", "v1"])
        with self.assertRaises(ValueError):
            training_v2.fit_low_rank_basis(self.meta, self.labels, self.masks, ["t1", "v1"], 1)
        with self.assertRaises(ValueError):
            training_v2.build_fc_anchor(self.meta, self.labels, self.masks, ["t1", "v1"])
        with self.assertRaises(ValueError):
            training_v2.estimate_high_effect_weights(self.meta, self.labels, self.masks, ["t1", "v1"])

    def test_cv_statistics_require_explicit_fold_train_ids(self):
        with self.assertRaises((TypeError, ValueError)):
            training_v2.fit_cv_fold_statistics(self.meta, self.labels, self.masks, None, 1)
        stats = training_v2.fit_cv_fold_statistics(self.meta, self.labels, self.masks, ["t1", "t2"], 1)
        self.assertEqual(stats["protein_basis"].shape, (1, 2))

    def test_control_anchor_ignores_treatment_labels_but_tracks_controls(self):
        meta = pd.DataFrame({
            "sample_ID": ["c1", "c2", "t1"],
            "split_final": ["train", "train", "train"],
            "perturbation_no_concentration": ["Water", "DMSO", "DrugA"],
        }).set_index("sample_ID", drop=False)
        labels = pd.DataFrame([[1., 2.], [3., 4.], [10., 20.]], index=meta.index)
        masks = pd.DataFrame(True, index=meta.index, columns=[0, 1])
        original, _ = training_v2.fit_control_protein_anchor(meta, labels, masks, ["c1", "c2"])
        treatment_changed = labels.copy(); treatment_changed.loc["t1"] = [1e6, -1e6]
        unchanged, _ = training_v2.fit_control_protein_anchor(meta, treatment_changed, masks, ["c1", "c2"])
        self.assertTrue(np.array_equal(original, unchanged))
        control_changed = labels.copy(); control_changed.loc["c1"] = [101., 202.]
        changed, _ = training_v2.fit_control_protein_anchor(meta, control_changed, masks, ["c1", "c2"])
        self.assertFalse(np.array_equal(original, changed))

    def test_chemical_variants_require_no_protein_labels(self):
        signature = inspect.signature(training_v2.make_chemical_feature_variant)
        self.assertEqual(tuple(signature.parameters), ("artifacts", "mode", "seed"))
        artifacts = training_v2.load_artifact_bundle()
        with patch.object(training_v2, "load_label_frames", side_effect=AssertionError("labels must not be read")):
            values, masks, order, _, _ = training_v2.make_chemical_feature_variant(artifacts, "shuffle", 20260814)
        self.assertEqual(values.shape, masks.shape)
        self.assertEqual(len(order), values.shape[0])

    def test_validation_macro_never_calls_backward(self):
        class EvalOnly(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1.0))
            def forward(self, batch):
                pred = self.weight * torch.ones(2, 2)
                return {"y_pred": pred, "delta_batch": pred * 0}
        dummy = V2Batch(*([torch.empty(2, 0)] * 15))
        loader = [(dummy, torch.zeros(2, 2), torch.ones(2, 2, dtype=torch.bool))]
        model = EvalOnly()
        macro, scores = training_v2.validation_macro_huber(model, {name: loader for name in training_v2.VAL_SCENARIOS}, LossWeights(batch_reg=0.0))
        self.assertIsNone(model.weight.grad)
        self.assertEqual(set(scores), set(training_v2.VAL_SCENARIOS))
        self.assertTrue(np.isfinite(macro))


if __name__ == "__main__":
    unittest.main()
