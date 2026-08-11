import unittest

import numpy as np
import pandas as pd
import torch

from baseline.evaluation import build_matched_control_pairs, compute_fold_change
from aivc.losses import (
    compute_target_edge_corr,
    correlation_consistency_loss,
    fc_pearson_loss,
    residual_l2_loss,
)


MATCH_DEFAULTS = {
    "data_source": "WAYB",
    "instrument": "Orbitrap",
    "Yeast_cell_plate": "plate_A",
    "Strains": "S1",
    "Medium": "YPD",
    "Temperature": 30,
    "pert_time": 60,
}


def metadata_row(sample_id, perturbation, split="train", **overrides):
    row = {
        "sample_ID": sample_id,
        "split_final": split,
        "perturbation_no_concentration": perturbation,
        **MATCH_DEFAULTS,
    }
    row.update(overrides)
    return row


class FoldChangeTargetTests(unittest.TestCase):
    def setUp(self):
        rows = [
            metadata_row("ctrl_1", "Water"),
            metadata_row("ctrl_2", "DMSO"),
            metadata_row("treat_matched", "DrugA"),
            metadata_row("treat_unmatched", "DrugB", Yeast_cell_plate="plate_B"),
            metadata_row("val_ctrl", "Water", split="val_both", Yeast_cell_plate="plate_B"),
            metadata_row("qc", "QC_pool"),
        ]
        self.meta = pd.DataFrame(rows).set_index("sample_ID")
        self.y = pd.DataFrame(
            [
                [10.0, np.nan, 30.0],
                [12.0, 20.0, np.nan],
                [15.0, 25.0, np.nan],
                [18.0, 24.0, 32.0],
                [17.0, 22.0, 31.0],
                [11.0, 21.0, 31.0],
            ],
            index=self.meta.index,
            columns=["P1", "P2", "P3"],
        )
        self.mask = self.y.notna()
        self.train_mask = self.meta["split_final"].eq("train")

    def test_pairs_are_exact_train_only_and_exclude_qc(self):
        pairs = build_matched_control_pairs(self.meta, self.train_mask)
        self.assertEqual(pairs["treatment_sample_ID"].tolist(), ["treat_matched"])
        self.assertEqual(pairs.iloc[0]["control_sample_ids"], ("ctrl_1", "ctrl_2"))
        self.assertEqual(pairs.iloc[0]["n_controls"], 2)

    def test_fold_change_averages_controls_protein_by_protein(self):
        result = compute_fold_change(
            self.meta, self.y, self.mask, train_mask=self.train_mask
        )
        fc = result["fc_true"].loc["treat_matched"]
        fc_mask = result["fc_mask"].loc["treat_matched"]

        # P1 control mean=(10+12)/2=11, treatment=15, so FC=4.
        self.assertAlmostEqual(fc["P1"], 4.0)
        # P2 has only one observed control (20), so FC=25-20=5.
        self.assertAlmostEqual(fc["P2"], 5.0)
        # Treatment P3 is missing, so it must not contribute to FC loss.
        self.assertTrue(np.isnan(fc["P3"]))
        self.assertEqual(fc_mask.tolist(), [True, True, False])


class LossTests(unittest.TestCase):
    def test_fc_pearson_perfect_and_inverse(self):
        true = torch.tensor([[1.0, 2.0, 3.0, 999.0]])
        mask = torch.tensor([[True, True, True, False]])
        perfect = torch.tensor([[2.0, 4.0, 6.0, -100.0]], requires_grad=True)
        inverse = torch.tensor([[-1.0, -2.0, -3.0, -100.0]])

        self.assertAlmostEqual(fc_pearson_loss(perfect, true, mask).item(), 0.0, places=6)
        self.assertAlmostEqual(fc_pearson_loss(inverse, true, mask).item(), 2.0, places=6)

    def test_fc_pearson_undefined_batch_is_skipped_and_differentiable(self):
        pred = torch.tensor([[1.0]], requires_grad=True)
        loss = fc_pearson_loss(pred, torch.tensor([[2.0]]), torch.tensor([[True]]))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_residual_l2_has_interpretable_scale(self):
        pred_dict = {
            "delta_drug": torch.ones(2, 3),
            "delta_strain": torch.full((2, 3), 2.0),
            "delta_context": torch.zeros(2, 3),
        }
        # Mean of head-wise mean squares: (1 + 4 + 0) / 3.
        self.assertAlmostEqual(residual_l2_loss(pred_dict).item(), 5.0 / 3.0, places=6)

    def test_sparse_correlation_consistency(self):
        base = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        pred = torch.stack([base, 2.0 * base, -base], dim=1).requires_grad_()
        edge_index = torch.tensor([[0, 0], [1, 2]])
        target_corr = torch.tensor([1.0, -1.0])

        loss = correlation_consistency_loss(pred, edge_index, target_corr)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_target_edge_corr_consumes_b_edges_and_uses_pairwise_rows(self):
        # B supplies only the selected edge list.  C computes the corresponding
        # training target on jointly observed rows; the unmatched extreme value
        # in P0 must not change either correlation.
        values = torch.tensor(
            [
                [0.0, 0.0, 2.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 0.0],
                [100.0, float("nan"), float("nan")],
            ]
        )
        mask = torch.isfinite(values)
        filled = torch.nan_to_num(values, nan=0.0).requires_grad_()
        b_edge_index = torch.tensor([[0, 0], [1, 2]])

        target = compute_target_edge_corr(
            filled, b_edge_index, mask=mask, min_pairs=3
        )

        self.assertTrue(torch.allclose(target, torch.tensor([1.0, -1.0]), atol=1e-6))
        self.assertFalse(target.requires_grad)

if __name__ == "__main__":
    unittest.main()
