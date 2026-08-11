"""Person C5 loss-ablation orchestration framework.

This module owns experiment planning, result contracts and artifact logging.
It does not construct B4's protein graph.  The full correlation experiment is
marked as waiting until B provides the final ``edge_index`` and C3 computes the
strictly aligned ``target_edge_corr`` from training rows only.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from aivc.config import get_experiment_config
from baseline.evaluation import TEST_SPLITS, VAL_SPLITS


LOSS_ABLATION_ORDER = (
    "mse_only",
    "mse_fc",
    "mse_fc_l2",
    "full_multitask",
)


def get_loss_ablation_configs(shared_config=None):
    """Return comparable C5 loss configurations in fixed experiment order."""
    config = get_experiment_config(shared_config)
    configured = config["loss"]
    base = {
        "mse": float(configured["mse_weight"]),
        "fc": float(configured["fc_pearson_weight"]),
        "l2": float(configured["residual_l2_weight"]),
        "corr": float(configured["correlation_consistency_weight"]),
    }
    definitions = {
        "mse_only": {
            "description": "仅 mask-aware MSE",
            "loss_weights": {"mse": base["mse"], "fc": 0.0, "l2": 0.0, "corr": 0.0},
            "requires_correlation_bundle": False,
        },
        "mse_fc": {
            "description": "MSE + FC Pearson",
            "loss_weights": {"mse": base["mse"], "fc": base["fc"], "l2": 0.0, "corr": 0.0},
            "requires_correlation_bundle": False,
        },
        "mse_fc_l2": {
            "description": "MSE + FC Pearson + 残差 L2",
            "loss_weights": {"mse": base["mse"], "fc": base["fc"], "l2": base["l2"], "corr": 0.0},
            "requires_correlation_bundle": False,
        },
        "full_multitask": {
            "description": "MSE + FC Pearson + 残差 L2 + 相关一致性",
            "loss_weights": base,
            "requires_correlation_bundle": True,
        },
    }
    return {name: deepcopy(definitions[name]) for name in LOSS_ABLATION_ORDER}


def validate_correlation_bundle(bundle, n_proteins):
    """Validate the B4-edge/C3-target hand-off without changing B's graph."""
    if bundle is None:
        raise ValueError("correlation_bundle is required")
    if "edge_index" not in bundle or "target_edge_corr" not in bundle:
        raise KeyError("correlation_bundle requires edge_index and target_edge_corr")
    edge_index = bundle["edge_index"]
    target = bundle["target_edge_corr"]
    if not torch.is_tensor(edge_index) or not torch.is_tensor(target):
        raise TypeError("edge_index and target_edge_corr must be torch tensors")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edges)")
    if target.ndim != 1 or target.shape[0] != edge_index.shape[1]:
        raise ValueError("target_edge_corr must align one-to-one with edge_index")
    if edge_index.numel() and (
        int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= int(n_proteins)
    ):
        raise ValueError("edge_index contains a protein index outside the retained columns")
    if target.numel() and not torch.isfinite(target).any():
        raise ValueError("target_edge_corr contains no usable correlation targets")
    return True


def build_experiment_plan(n_proteins, correlation_bundle=None, shared_config=None):
    """Build the C5 plan while exposing the unresolved B4 dependency."""
    configs = get_loss_ablation_configs(shared_config)
    bundle_ready = correlation_bundle is not None
    if bundle_ready:
        validate_correlation_bundle(correlation_bundle, n_proteins)

    plan = []
    for name, config in configs.items():
        waiting = config["requires_correlation_bundle"] and not bundle_ready
        plan.append(
            {
                "name": name,
                "description": config["description"],
                "loss_weights": config["loss_weights"],
                "requires_correlation_bundle": config["requires_correlation_bundle"],
                "status": "waiting_b4_edge_index" if waiting else "ready",
            }
        )
    return plan


def evaluate_prediction_metrics(y_true, y_pred, mask):
    """Compute the common mask-aware metrics recorded for every C5 split."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError("y_true, y_pred and mask must have identical shapes")
    valid = mask & np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        return {"log2_rmse": None, "global_r2": None, "per_protein_r2_median": None}

    residual = y_true[valid] - y_pred[valid]
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    observed = y_true[valid]
    ss_tot = float(np.square(observed - observed.mean()).sum())
    global_r2 = float(1.0 - np.square(residual).sum() / ss_tot) if ss_tot > 0 else None

    protein_scores = []
    for column in range(y_true.shape[1]):
        column_valid = valid[:, column]
        if int(column_valid.sum()) < 3:
            continue
        truth = y_true[column_valid, column]
        total = float(np.square(truth - truth.mean()).sum())
        if total <= 0:
            continue
        error = float(np.square(truth - y_pred[column_valid, column]).sum())
        protein_scores.append(1.0 - error / total)
    median = float(np.median(protein_scores)) if protein_scores else None
    return {
        "log2_rmse": rmse,
        "global_r2": global_r2,
        "per_protein_r2_median": median,
    }


def validate_experiment_result(result):
    """Enforce the C5-to-C6 result contract before anything is recorded."""
    required = {"history", "metrics", "baseline_comparison", "submission_checks", "artifacts"}
    missing = sorted(required.difference(result))
    if missing:
        raise KeyError(f"Experiment result is missing fields: {missing}")
    unknown_splits = sorted(set(result["metrics"]).difference(VAL_SPLITS + TEST_SPLITS))
    if unknown_splits:
        raise ValueError(f"Unknown evaluation splits: {unknown_splits}")
    return True


def run_experiment_plan(
    experiment_runner,
    n_proteins,
    output_dir,
    correlation_bundle=None,
    shared_config=None,
):
    """Run ready C5 configurations through an injected end-to-end runner.

    ``experiment_runner`` owns the concrete train/validation/submission work and
    receives ``(plan_item, correlation_bundle)``.  This separation lets the C5
    framework be completed now while the full correlation experiment remains
    blocked on B4's final edge list.
    """
    plan = build_experiment_plan(n_proteins, correlation_bundle, shared_config)
    records = []
    for item in plan:
        if item["status"] != "ready":
            records.append({**item, "result": None})
            continue
        result = experiment_runner(deepcopy(item), correlation_bundle)
        validate_experiment_result(result)
        records.append({**item, "status": "completed", "result": result})

    payload = {
        "schema_version": 1,
        "data_discipline": {
            "fit_statistics": "train_only",
            "validation_used_for_training": False,
            "test_truth_used_for_training": False,
            "early_stopping_split": "val_both",
            "prediction_scale": "log2",
        },
        "records": records,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "loss_ablation_results.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload, output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect the Person C5 loss experiment plan")
    parser.add_argument("--n-proteins", type=int, default=4422)
    parser.add_argument("--output", type=Path, default=None, help="Optional plan JSON path")
    args = parser.parse_args(argv)
    plan = build_experiment_plan(args.n_proteins)
    text = json.dumps(plan, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
