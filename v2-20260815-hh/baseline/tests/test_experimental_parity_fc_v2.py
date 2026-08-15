import inspect
import json
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.baseline import training_v2
from baseline.baseline.chemical_ood_analysis_v2 import _build_model
from baseline.baseline.losses_v2 import (
    LossWeights,
    compute_losses,
    masked_rowwise_pearson_loss,
)


ROOT = Path(__file__).resolve().parents[2]
EXACT_KEYS = [
    "data_source", "Strains", "Medium", "Temperature", "pert_time",
    "pert_time_unit", "instrument", "Yeast_cell_plate",
]


class ExperimentalParityFCLossTests(unittest.TestCase):
    def test_rowwise_pearson_and_duplicate_rows_keep_row_mean(self):
        pred = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], requires_grad=True)
        target = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        mask = torch.ones_like(pred, dtype=torch.bool)
        loss = masked_rowwise_pearson_loss(pred, target, mask)
        duplicate = masked_rowwise_pearson_loss(
            pred.repeat(3, 1), target.repeat(3, 1), mask.repeat(3, 1),
        )
        self.assertAlmostEqual(float(loss.detach()), 1.0, places=6)
        self.assertAlmostEqual(float(duplicate.detach()), float(loss.detach()), places=6)

    def test_rowwise_joint_mask_and_masked_values(self):
        pred = torch.tensor([[1.0, 2.0, 3.0, 999.0]])
        target = torch.tensor([[1.0, 2.0, 3.0, -999.0]])
        mask = torch.tensor([[True, True, True, False]])
        reference = masked_rowwise_pearson_loss(pred, target, mask)
        changed = masked_rowwise_pearson_loss(
            pred + torch.tensor([[0.0, 0.0, 0.0, 1e8]]), target, mask,
        )
        self.assertAlmostEqual(float(reference), 0.0, places=6)
        self.assertAlmostEqual(float(changed), float(reference), places=6)

    def test_empty_or_constant_fc_rows_return_differentiable_zero(self):
        pred = torch.ones((2, 3), requires_grad=True)
        target = torch.ones((2, 3))
        mask = torch.tensor([[False, False, False], [True, True, True]])
        loss = masked_rowwise_pearson_loss(pred, target, mask)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))

    def test_total_loss_is_huber_plus_point_one_rowwise_fc(self):
        y_pred = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0]], requires_grad=True)
        outputs = {"y_pred": y_pred, "delta_batch": torch.zeros_like(y_pred)}
        fc_pred = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], requires_grad=True)
        fc_true = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        mask = torch.ones_like(y_pred, dtype=torch.bool)
        weights = LossWeights(absolute=1.0, fc=0.1, dep=0, pathway=0, batch_reg=0)
        result = compute_losses(outputs, torch.zeros_like(y_pred), mask, weights, fc_pred, fc_true, mask)
        expected = result["loss_absolute"] + 0.1 * result["loss_fc"]
        self.assertTrue(torch.allclose(result["loss_total"], expected, atol=0, rtol=0))


class ExperimentalParityFCMappingTests(unittest.TestCase):
    def synthetic_frames(self):
        common = {
            "data_source": "A", "Strains": "S", "Medium": "M", "Temperature": 30,
            "pert_time": 60, "pert_time_unit": "min", "instrument": "I", "Yeast_cell_plate": "P",
        }
        rows = [
            {"sample_ID": "t", "split_final": "train", "perturbation_no_concentration": "Drug", "pert_id": "#3", **common},
            {"sample_ID": "w1", "split_final": "train", "perturbation_no_concentration": "Water", "pert_id": "#1", **common},
            {"sample_ID": "w2", "split_final": "train", "perturbation_no_concentration": "Water", "pert_id": "#1", **common},
            {"sample_ID": "d", "split_final": "train", "perturbation_no_concentration": "DMSO", "pert_id": "#2", **common},
            {"sample_ID": "v", "split_final": "val_chem_only", "perturbation_no_concentration": "Drug V", "pert_id": "#5", **common},
        ]
        meta = pd.DataFrame(rows).set_index("sample_ID", drop=False)
        labels = pd.DataFrame(
            [[10, 20, 30], [2, 4, 8], [4, 0, 10], [100, 100, 100], [9, 9, 9]],
            index=meta.index, columns=["p1", "p2", "p3"], dtype=np.float32,
        )
        masks = pd.DataFrame(
            [[1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1]],
            index=meta.index, columns=labels.columns, dtype=bool,
        )
        return meta, labels, masks

    def test_exact_water_match_and_proteinwise_control_mean(self):
        meta, labels, masks = self.synthetic_frames()
        result = training_v2.build_experimental_parity_fc_targets(meta, labels, masks, ["t"])
        np.testing.assert_allclose(result.fc_true[0], [7.0, 16.0, 21.0])
        np.testing.assert_array_equal(result.fc_mask[0], [True, True, True])
        self.assertEqual(result.pairing_table.loc[0, "matched_control_count"], 2)
        self.assertEqual(result.audit["matched_treatment_counts_by_control"]["Water"], 1)
        self.assertFalse(result.audit["fallback_used"])

    def test_validation_treatment_is_rejected(self):
        meta, labels, masks = self.synthetic_frames()
        with self.assertRaisesRegex(ValueError, "fold-train"):
            training_v2.build_experimental_parity_fc_targets(meta, labels, masks, ["v"])

    def test_frozen_spec_and_config_are_explicitly_nonofficial(self):
        spec = json.loads((ROOT / "project_v2/data_contract/control_matching_spec.json").read_text(encoding="utf-8"))
        config = json.loads((ROOT / "baseline/configs/model_v2_experimental_parity_fc_morgan_no_batch.yaml").read_text(encoding="utf-8"))
        self.assertEqual(spec["exact_match_keys"], EXACT_KEYS)
        self.assertTrue(config["experimental_nonofficial_parity_fc"])
        self.assertFalse(config["official_fc_result"])
        self.assertEqual(config["control_mapping"]["name"], "pert_id_parity_v1")
        self.assertEqual(config["loss"]["fc_weight"], 0.1)
        self.assertEqual(config["model"]["chemical_feature_components"], "morgan_only")
        self.assertFalse(config["model"]["batch_enabled"])

    def test_real_train_pair_prediction_identity(self):
        artifacts = training_v2.load_artifact_bundle(ROOT)
        meta = training_v2.load_train_val_metadata(ROOT)
        labels, masks = training_v2.load_label_frames(meta, artifacts, ROOT)
        train_ids = meta.index[meta["split_final"].eq("train")]
        names = meta.loc[train_ids, "perturbation_no_concentration"].astype(str).str.lower()
        treatment_ids = train_ids[~names.isin(training_v2.CONTROL_NAMES | {"quality control"})][:128]
        targets = training_v2.build_experimental_parity_fc_targets(meta, labels, masks, treatment_ids)
        self.assertGreater(targets.audit["matched_treatment_count"], 0)
        vocab = training_v2.fit_category_vocabulary(meta, train_ids)
        payload = training_v2.load_checkpoint(
            ROOT / "reports/model_v2_stage2/formal_huber_no_batch_seed_20260814/stage_a_best.pt",
            artifacts.hashes,
        )
        model = _build_model(payload, artifacts, vocab, "cpu")
        variant = training_v2.make_chemical_feature_variant(artifacts, "correct", 20260814)
        audit = training_v2.validate_real_matched_pair_prediction_identity(
            model, meta, targets.pairing_table, artifacts, vocab, variant,
            "correct", "morgan_only", "correct", 20260814,
        )
        self.assertLessEqual(audit["max_abs_identity_error"], 1e-5)
        self.assertEqual(audit["control_delta_response_max_abs"], 0.0)

    def test_read_only_analysis_requires_explicit_nonofficial_fc_opt_in(self):
        artifacts = training_v2.load_artifact_bundle(ROOT)
        meta = training_v2.load_train_val_metadata(ROOT)
        train_ids = meta.index[meta["split_final"].eq("train")]
        vocab = training_v2.fit_category_vocabulary(meta, train_ids)
        payload = deepcopy(training_v2.load_checkpoint(
            ROOT / "reports/model_v2_stage2/formal_huber_no_batch_seed_20260814/stage_a_best.pt",
            artifacts.hashes,
        ))
        payload["config"]["experimental_nonofficial_parity_fc"] = True
        payload["config"]["official_fc_result"] = False
        payload["config"]["control_mapping"] = {"name": "pert_id_parity_v1"}
        payload["config"]["loss"]["fc_weight"] = 0.1
        with self.assertRaisesRegex(ValueError, "Huber-only"):
            _build_model(payload, artifacts, vocab, "cpu")
        model = _build_model(
            payload,
            artifacts,
            vocab,
            "cpu",
            allow_experimental_nonofficial_parity_fc=True,
        )
        self.assertFalse(model.cfg.batch_enabled)

    def test_training_source_contains_no_test_proteome_path(self):
        source = inspect.getsource(training_v2)
        forbidden = "proteome" + "_raw_test"
        self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
