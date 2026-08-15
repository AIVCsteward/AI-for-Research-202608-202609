from __future__ import annotations

import unittest

import torch

from baseline.baseline.losses_v2 import (
    LossWeights, batch_regularization, compute_losses, masked_huber,
    masked_pearson_loss, paired_fold_change,
)


class ModelV2LossTests(unittest.TestCase):
    def test_huber_is_strictly_mask_aware_and_fill_invariant(self):
        pred = torch.tensor([[0.0, 50.0], [2.0, -50.0]], requires_grad=True)
        mask = torch.tensor([[True, False], [True, False]])
        first = masked_huber(pred, torch.tensor([[1.0, -999.0], [4.0, 999.0]]), mask)
        second = masked_huber(pred, torch.tensor([[1.0, 1e9], [4.0, -1e9]]), mask)
        self.assertTrue(torch.equal(first, second))
        first.backward()
        self.assertEqual(float(pred.grad[:, 1].abs().sum()), 0.0)

    def test_empty_mask_returns_safe_differentiable_zero(self):
        pred = torch.randn(2, 3, requires_grad=True)
        loss = masked_huber(pred, torch.full((2, 3), float("nan")), torch.zeros(2, 3, dtype=torch.bool))
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))

    def test_fc_uses_treatment_control_joint_mask(self):
        treatment = torch.tensor([[5.0, 9.0, 8.0, 6.0]])
        control = torch.tensor([[2.0, 3.0, 4.0, 1.0]])
        treatment_mask = torch.tensor([[True, True, False, True]])
        control_mask = torch.tensor([[True, False, True, True]])
        fc, joint = paired_fold_change(treatment, control, treatment_mask, control_mask)
        self.assertEqual(joint.tolist(), [[True, False, False, True]])
        self.assertEqual(fc.tolist(), [[3.0, 0.0, 0.0, 5.0]])
        self.assertAlmostEqual(float(masked_pearson_loss(fc, fc, joint)), 0.0, places=6)

    def test_total_equals_all_weighted_terms(self):
        outputs = {"y_pred": torch.ones(2, 3), "delta_batch": torch.full((2, 3), 2.0)}
        weights = LossWeights(absolute=1.2, fc=0.4, dep=0.0, pathway=0.0, batch_reg=0.3, batch_center_strength=0.5)
        fc_pred = torch.tensor([[1.0, 2.0, 3.0]])
        fc_true = torch.tensor([[1.0, 2.0, 4.0]])
        parts = compute_losses(outputs, torch.zeros(2, 3), torch.ones(2, 3, dtype=torch.bool), weights, fc_pred, fc_true, torch.ones(1, 3, dtype=torch.bool))
        expected = weights.absolute * parts["loss_absolute"] + weights.fc * parts["loss_fc"] + weights.dep * parts["loss_dep"] + weights.pathway * parts["loss_pathway"] + weights.batch_reg * parts["loss_batch_reg"]
        self.assertTrue(torch.allclose(parts["loss_total"], expected))
        self.assertGreater(float(batch_regularization(outputs["delta_batch"], 1.0)), float(outputs["delta_batch"].square().mean()))

    def test_batch_center_strength_is_internal_to_batch_regularization(self):
        outputs = {"y_pred": torch.zeros(2, 2), "delta_batch": torch.tensor([[1.0, 3.0], [1.0, 3.0]])}
        weights = LossWeights(absolute=0.0, batch_reg=1e-4, batch_center_strength=1.0)
        parts = compute_losses(outputs, torch.zeros(2, 2), torch.ones(2, 2, dtype=torch.bool), weights)
        expected_batch = outputs["delta_batch"].square().mean() + outputs["delta_batch"].mean(0).square().mean()
        self.assertTrue(torch.equal(parts["loss_batch_reg"], expected_batch))
        self.assertTrue(torch.equal(parts["loss_total"], 1e-4 * expected_batch))

    def test_zero_weight_terms_do_not_affect_total_or_require_fc(self):
        outputs = {"y_pred": torch.ones(1, 2), "delta_batch": torch.randn(1, 2)}
        mask = torch.ones(1, 2, dtype=torch.bool)
        weights = LossWeights(absolute=1.0, fc=0.0, dep=0.0, pathway=0.0, batch_reg=0.0)
        parts = compute_losses(outputs, torch.zeros(1, 2), mask, weights)
        self.assertTrue(torch.equal(parts["loss_total"], parts["loss_absolute"]))


if __name__ == "__main__":
    unittest.main()
