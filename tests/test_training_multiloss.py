import unittest
from unittest.mock import patch

import pandas as pd
import torch
from torch import nn

from aivc.training import (
    compute_multitask_batch_loss,
    masked_per_protein_r2_median,
    prepare_fold_change_index,
    train,
)


class TinyResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        y_raw = x * self.scale
        baseline = 0.4 * y_raw
        delta_drug = 0.1 * y_raw
        delta_strain = 0.2 * y_raw
        delta_context = 0.3 * y_raw
        # Deliberately make y_pred different: FC must still be computed from y_raw.
        y_pred = -y_raw
        return {
            "y_pred": y_pred,
            "y_raw": y_raw,
            "baseline": baseline,
            "delta_drug": delta_drug,
            "delta_strain": delta_strain,
            "delta_context": delta_context,
            "delta_cal": y_pred - y_raw,
        }


class FoldChangeTrainingTests(unittest.TestCase):
    def setUp(self):
        self.sample_ids = ["treatment", "control", "other"]
        self.pairs = pd.DataFrame(
            {
                "treatment_sample_ID": ["treatment"],
                "control_sample_ids": [("control",)],
                "n_controls": [1],
            }
        )

    def test_prepare_fold_change_index_uses_train_row_positions(self):
        index = prepare_fold_change_index(self.sample_ids, self.pairs)
        self.assertEqual(index.shape, (3, 1))
        self.assertEqual(index[:, 0].tolist(), [1, -1, -1])

    def test_fc_loss_uses_treatment_minus_control_y_raw(self):
        model = TinyResidualModel()
        X = torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])
        # The true treatment-control FC is also [1, 2].
        y = torch.tensor([[11.0, 22.0], [10.0, 20.0], [0.0, 0.0]])
        mask = torch.ones_like(y)
        control_index = prepare_fold_change_index(self.sample_ids, self.pairs)

        _, components, _ = compute_multitask_batch_loss(
            model,
            X,
            y,
            mask,
            torch.tensor([0]),
            fc_control_index=control_index,
            loss_weights={"mse": 0.0, "fc": 1.0, "l2": 0.0, "corr": 0.0},
        )
        self.assertAlmostEqual(components["loss_fc"].item(), 0.0, places=6)

    def test_train_records_named_loss_components(self):
        model = TinyResidualModel()
        X = torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])
        y = -X.clone()
        mask = torch.ones_like(y)
        control_index = prepare_fold_change_index(self.sample_ids, self.pairs)
        val_data = {
            "val_both": {"X": X, "y_filled": y, "mask_t": mask},
        }

        _, history = train(
            model,
            X,
            y,
            mask,
            val_data,
            epochs=2,
            batch_size=3,
            lr=1e-2,
            verbose=False,
            fc_control_index=control_index,
            loss_weights={"corr": 0.0},
            early_stopping_patience=2,
        )
        for key in ("loss_total", "loss_mse", "loss_fc", "loss_l2", "loss_corr"):
            self.assertEqual(len(history[key]), 2)
        self.assertEqual(len(history["val_per_protein_r2"]["val_both"]), 2)

    def test_nonzero_correlation_loss_reaches_batch_training(self):
        model = TinyResidualModel()
        X = torch.tensor(
            [[1.0, 1.0], [2.0, 0.0], [3.0, -1.0]], dtype=torch.float32
        )
        y = torch.zeros_like(X)
        mask = torch.ones_like(X)
        edge_index = torch.tensor([[0], [1]])
        target_edge_corr = torch.tensor([0.0])

        loss, components, _ = compute_multitask_batch_loss(
            model,
            X,
            y,
            mask,
            torch.arange(3),
            edge_index=edge_index,
            target_edge_corr=target_edge_corr,
            loss_weights={"mse": 0.0, "fc": 0.0, "l2": 0.0, "corr": 1.0},
        )

        self.assertGreater(components["loss_corr"].item(), 0.0)
        self.assertAlmostEqual(loss.item(), components["loss_corr"].item(), places=6)

    def test_explicit_nonzero_corr_requires_b_and_c_inputs(self):
        model = TinyResidualModel()
        X = torch.tensor([[1.0, 1.0], [2.0, 0.0], [3.0, -1.0]])
        with self.assertRaises(ValueError):
            compute_multitask_batch_loss(
                model,
                X,
                torch.zeros_like(X),
                torch.ones_like(X),
                torch.arange(3),
                loss_weights={"mse": 1.0, "fc": 0.0, "l2": 0.0, "corr": 0.1},
            )

    def test_legacy_default_training_remains_graph_optional(self):
        model = TinyResidualModel()
        X = torch.tensor([[1.0, 1.0], [2.0, 0.0], [3.0, -1.0]])
        _, components, _ = compute_multitask_batch_loss(
            model,
            X,
            torch.zeros_like(X),
            torch.ones_like(X),
            torch.arange(3),
        )
        self.assertEqual(components["loss_corr"].item(), 0.0)

    def test_total_loss_equals_all_four_weighted_components(self):
        model = TinyResidualModel()
        sample_ids = ["treatment", "control", "other_1", "other_2"]
        pairs = pd.DataFrame(
            {
                "treatment_sample_ID": ["treatment"],
                "control_sample_ids": [("control",)],
                "n_controls": [1],
            }
        )
        control_index = prepare_fold_change_index(sample_ids, pairs)
        X = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, -1.0, 1.0],
            ]
        )
        y = torch.tensor(
            [
                [-1.0, -2.0, -3.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        mask = torch.ones_like(y)
        edge_index = torch.tensor([[0], [1]])
        target_edge_corr = torch.tensor([0.0])
        weights = {"mse": 1.7, "fc": 0.4, "l2": 0.2, "corr": 0.6}

        loss, components, _ = compute_multitask_batch_loss(
            model,
            X,
            y,
            mask,
            torch.tensor([0, 2, 3]),
            fc_control_index=control_index,
            edge_index=edge_index,
            target_edge_corr=target_edge_corr,
            loss_weights=weights,
        )
        for key in ("loss_mse", "loss_fc", "loss_l2", "loss_corr"):
            self.assertGreater(components[key].item(), 0.0)
        expected = (
            weights["mse"] * components["loss_mse"]
            + weights["fc"] * components["loss_fc"]
            + weights["l2"] * components["loss_l2"]
            + weights["corr"] * components["loss_corr"]
        )
        self.assertTrue(torch.allclose(loss, expected, atol=1e-7))

    def test_per_protein_r2_uses_numpy_style_even_median(self):
        y = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        pred = torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        mask = torch.ones_like(y)

        metric = masked_per_protein_r2_median(pred, y, mask)

        # Per-protein R2 values are 1 and 0, whose NumPy-style median is 0.5.
        self.assertAlmostEqual(metric.item(), 0.5, places=6)

    def test_early_stopping_restores_best_epoch(self):
        X = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
        y = torch.zeros_like(X)
        mask = torch.ones_like(X)
        val_data = {"val_both": {"X": X, "y_filled": y, "mask_t": mask}}
        weights = {"mse": 1.0, "fc": 0.0, "l2": 0.0, "corr": 0.0}

        one_epoch_model = TinyResidualModel()
        with patch(
            "aivc.training.masked_per_protein_r2_median",
            return_value=torch.tensor(0.5),
        ):
            one_epoch_model, _ = train(
                one_epoch_model,
                X,
                y,
                mask,
                val_data,
                epochs=1,
                batch_size=3,
                lr=1e-2,
                verbose=False,
                loss_weights=weights,
                early_stopping_patience=2,
            )

        stopped_model = TinyResidualModel()
        monitor_values = [torch.tensor(0.5), torch.tensor(0.4), torch.tensor(0.3)]
        with patch(
            "aivc.training.masked_per_protein_r2_median",
            side_effect=monitor_values,
        ):
            stopped_model, history = train(
                stopped_model,
                X,
                y,
                mask,
                val_data,
                epochs=10,
                batch_size=3,
                lr=1e-2,
                verbose=False,
                loss_weights=weights,
                early_stopping_patience=2,
            )

        self.assertEqual(len(history["monitor"]), 3)
        for actual, expected in zip(history["monitor"], [0.5, 0.4, 0.3]):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertTrue(
            torch.allclose(stopped_model.scale, one_epoch_model.scale, atol=1e-8)
        )


if __name__ == "__main__":
    unittest.main()
