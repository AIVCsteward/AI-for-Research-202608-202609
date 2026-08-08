import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.ablation_loss import (
    build_experiment_plan,
    evaluate_prediction_metrics,
    run_experiment_plan,
    validate_correlation_bundle,
)
from experiments.report_results import render_experiment_log


def fake_experiment_runner(plan_item, correlation_bundle):
    del correlation_bundle
    return {
        "history": {
            "loss_total": [1.0],
            "loss_mse": [0.8],
            "loss_fc": [0.1],
            "loss_l2": [0.1],
            "loss_corr": [0.0],
            "monitor": [0.2],
        },
        "metrics": {
            "val_both": {
                "n_samples": 3,
                "log2_rmse": 0.5,
                "global_r2": 0.4,
                "per_protein_r2_median": 0.2,
                "fc_pearson": 0.3,
            }
        },
        "baseline_comparison": {
            "summary": f"{plan_item['name']} compared with Matched Control",
            "attribution": "framework test",
        },
        "submission_checks": {
            "n_samples": 2,
            "n_proteins": 3,
            "no_na": True,
            "no_inf": True,
            "prediction_scale": "log2",
        },
        "artifacts": {"checkpoint": "checkpoint.pt", "prediction": "prediction.csv"},
    }


class C5FrameworkTests(unittest.TestCase):
    def test_plan_waits_only_for_full_correlation_experiment(self):
        plan = build_experiment_plan(n_proteins=3)
        statuses = {item["name"]: item["status"] for item in plan}
        self.assertEqual(statuses["mse_only"], "ready")
        self.assertEqual(statuses["mse_fc"], "ready")
        self.assertEqual(statuses["mse_fc_l2"], "ready")
        self.assertEqual(statuses["full_multitask"], "waiting_b4_edge_index")

    def test_correlation_bundle_requires_one_target_per_b_edge(self):
        valid = {
            "edge_index": torch.tensor([[0, 1], [1, 2]]),
            "target_edge_corr": torch.tensor([0.8, -0.4]),
        }
        self.assertTrue(validate_correlation_bundle(valid, n_proteins=3))
        with self.assertRaises(ValueError):
            validate_correlation_bundle(
                {
                    "edge_index": valid["edge_index"],
                    "target_edge_corr": torch.tensor([0.8]),
                },
                n_proteins=3,
            )

    def test_prediction_metrics_are_mask_aware(self):
        y_true = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 999.0]])
        y_pred = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, -999.0]])
        mask = np.array([[True, True], [True, True], [True, False]])
        metrics = evaluate_prediction_metrics(y_true, y_pred, mask)
        self.assertAlmostEqual(metrics["log2_rmse"], 0.0, places=7)
        self.assertAlmostEqual(metrics["global_r2"], 1.0, places=7)
        self.assertAlmostEqual(metrics["per_protein_r2_median"], 1.0, places=7)

    def test_plan_runs_ready_items_and_c6_renders_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            payload, path = run_experiment_plan(
                fake_experiment_runner,
                n_proteins=3,
                output_dir=Path(temp_dir),
            )
            self.assertTrue(path.exists())
            persisted = json.loads(path.read_text(encoding="utf-8"))
            completed = [r for r in persisted["records"] if r["status"] == "completed"]
            waiting = [r for r in persisted["records"] if r["status"].startswith("waiting")]
            self.assertEqual(len(completed), 3)
            self.assertEqual(len(waiting), 1)

            markdown = render_experiment_log(payload)
            self.assertIn("mse_only", markdown)
            self.assertIn("waiting_b4_edge_index", markdown)
            self.assertIn("val_both", markdown)
            self.assertIn("prediction_scale", markdown)


if __name__ == "__main__":
    unittest.main()
