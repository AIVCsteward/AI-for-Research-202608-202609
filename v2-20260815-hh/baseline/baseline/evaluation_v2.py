"""Canonical V2 metrics, scenario evaluation and branch diagnostics."""
from __future__ import annotations

import numpy as np
import torch


VAL_SCENARIOS = ("val_chem_only", "val_strain_only", "val_both", "val_time")
ATTRIBUTION_OUTPUTS = ("full", "no_batch_posthoc", "batch_only", "baseline_only")


def _valid(y_true, y_pred, mask):
    return np.asarray(mask, bool) & np.isfinite(y_true) & np.isfinite(y_pred)


def masked_rmse(y_true, y_pred, mask):
    valid = _valid(y_true, y_pred, mask)
    return float(np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2))) if valid.any() else np.nan


def masked_mae(y_true, y_pred, mask):
    valid = _valid(y_true, y_pred, mask)
    return float(np.mean(np.abs(y_true[valid] - y_pred[valid]))) if valid.any() else np.nan


def masked_global_r2(y_true, y_pred, mask):
    valid = _valid(y_true, y_pred, mask)
    if valid.sum() < 2:
        return np.nan
    true = y_true[valid]
    denominator = np.square(true - true.mean()).sum()
    return float(1 - np.square(true - y_pred[valid]).sum() / denominator) if denominator > 0 else np.nan


def _paired_pcc(true, pred):
    left = true - true.mean()
    right = pred - pred.mean()
    denominator = np.sqrt(np.square(left).sum() * np.square(right).sum())
    return float(np.sum(left * right) / denominator) if denominator > 0 else np.nan


def _paired_r2(true, pred):
    denominator = np.square(true - true.mean()).sum()
    return float(1 - np.square(true - pred).sum() / denominator) if denominator > 0 else np.nan


def _median_axis_metric(y_true, y_pred, mask, axis, metric, min_count):
    values = []
    n_items = y_true.shape[axis]
    for item in range(n_items):
        if axis == 0:
            true_slice, pred_slice, mask_slice = y_true[item], y_pred[item], mask[item]
        else:
            true_slice, pred_slice, mask_slice = y_true[:, item], y_pred[:, item], mask[:, item]
        valid = _valid(true_slice, pred_slice, mask_slice)
        if valid.sum() < min_count:
            continue
        value = metric(true_slice[valid], pred_slice[valid])
        if np.isfinite(value):
            values.append(value)
    return float(np.median(values)) if values else np.nan


def median_per_sample_pcc(y_true, y_pred, mask, min_count=3):
    return _median_axis_metric(y_true, y_pred, mask, 0, _paired_pcc, min_count)


def median_per_sample_r2(y_true, y_pred, mask, min_count=3):
    return _median_axis_metric(y_true, y_pred, mask, 0, _paired_r2, min_count)


def median_per_protein_pcc(y_true, y_pred, mask, min_count=3):
    return _median_axis_metric(y_true, y_pred, mask, 1, _paired_pcc, min_count)


def median_per_protein_r2(y_true, y_pred, mask, min_count=3):
    return _median_axis_metric(y_true, y_pred, mask, 1, _paired_r2, min_count)


def batch_diagnostics(outputs):
    batch = outputs["delta_batch"].detach().float()
    response = outputs["delta_response"].detach().float()
    response_norm = torch.linalg.vector_norm(response)
    ratio = torch.linalg.vector_norm(batch) / response_norm.clamp_min(1e-12)
    return {
        "delta_batch_mean": float(batch.mean()),
        "delta_batch_std": float(batch.std(unbiased=False)),
        "delta_batch_max_abs": float(batch.abs().max()),
        "delta_batch_to_response_norm": float(ratio),
    }


def evaluate_batch_diagnostics(model, loader, device="cpu"):
    """Stream population diagnostics over a declared loader without retaining outputs."""
    total, total_square, count, maximum = 0.0, 0.0, 0, 0.0
    batch_norm_square, response_norm_square = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for batch, _, _ in loader:
            outputs = model(batch.to(device))
            delta_batch = outputs["delta_batch"].double()
            delta_response = outputs["delta_response"].double()
            total += float(delta_batch.sum())
            total_square += float(delta_batch.square().sum())
            count += delta_batch.numel()
            maximum = max(maximum, float(delta_batch.abs().max()))
            batch_norm_square += float(delta_batch.square().sum())
            response_norm_square += float(delta_response.square().sum())
    if count == 0:
        raise ValueError("batch diagnostic loader is empty")
    mean = total / count
    variance = max(total_square / count - mean * mean, 0.0)
    ratio = np.sqrt(batch_norm_square) / max(np.sqrt(response_norm_square), 1e-12)
    return {
        "delta_batch_mean": mean,
        "delta_batch_std": float(np.sqrt(variance)),
        "delta_batch_max_abs": maximum,
        "delta_batch_to_response_norm": float(ratio),
    }


def evaluate_component_norms(model, loaders, device="cpu"):
    """Return valid-position L2 norms for the three additive components."""
    if set(loaders) != set(VAL_SCENARIOS):
        raise ValueError("component norms require exactly four validation scenarios")
    result = {}
    model.eval()
    with torch.no_grad():
        for scenario in VAL_SCENARIOS:
            squares = {"y_baseline": 0.0, "delta_response": 0.0, "delta_batch": 0.0}
            for batch, _, mask in loaders[scenario]:
                outputs = model(batch.to(device))
                valid = mask.to(device).bool()
                for name in squares:
                    squares[name] += float(outputs[name][valid].double().square().sum())
            result[scenario] = {name: float(np.sqrt(value)) for name, value in squares.items()}
    return result


def evaluate_scenario(model, loader, device="cpu"):
    predictions, targets, masks = [], [], []
    model.eval()
    with torch.no_grad():
        for batch, target, mask in loader:
            batch = batch.to(device)
            outputs = model(batch)
            predictions.append(outputs["y_pred"].detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            masks.append(mask.detach().cpu().numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    mask = np.concatenate(masks)
    return {
        "rmse": masked_rmse(target, prediction, mask),
        "mae": masked_mae(target, prediction, mask),
        "global_r2": masked_global_r2(target, prediction, mask),
        "median_per_sample_pcc": median_per_sample_pcc(target, prediction, mask),
        "median_per_sample_r2": median_per_sample_r2(target, prediction, mask),
        "median_per_protein_pcc": median_per_protein_pcc(target, prediction, mask),
        "median_per_protein_r2": median_per_protein_r2(target, prediction, mask),
        "n_valid_positions": int(_valid(target, prediction, mask).sum()),
    }


def evaluate_four_scenarios(model, loaders, device="cpu"):
    missing = set(VAL_SCENARIOS) - set(loaders)
    if missing:
        raise ValueError(f"missing validation scenarios: {sorted(missing)}")
    return {name: evaluate_scenario(model, loaders[name], device=device) for name in VAL_SCENARIOS}


def evaluate_component_attribution(model, loaders, device="cpu"):
    """Post-hoc component attribution; never a substitute for a trained ablation."""
    if set(loaders) != set(VAL_SCENARIOS):
        raise ValueError("component attribution requires exactly four validation scenarios")
    result = {}
    model.eval()
    with torch.no_grad():
        for scenario in VAL_SCENARIOS:
            predictions = {name: [] for name in ATTRIBUTION_OUTPUTS}
            targets, masks = [], []
            norm_squares = {"y_baseline": 0.0, "delta_response": 0.0, "delta_batch": 0.0}
            for batch, target, mask in loaders[scenario]:
                outputs = model(batch.to(device))
                variants = {
                    "full": outputs["y_baseline"] + outputs["delta_response"] + outputs["delta_batch"],
                    "no_batch_posthoc": outputs["y_baseline"] + outputs["delta_response"],
                    "batch_only": outputs["y_baseline"] + outputs["delta_batch"],
                    "baseline_only": outputs["y_baseline"],
                }
                for name, prediction in variants.items():
                    predictions[name].append(prediction.detach().cpu().numpy())
                valid = mask.to(device).bool()
                for name in norm_squares:
                    component = outputs[name]
                    norm_squares[name] += float(component[valid].double().square().sum())
                targets.append(target.numpy())
                masks.append(mask.numpy())
            target = np.concatenate(targets)
            mask = np.concatenate(masks)
            metrics = {}
            for name in ATTRIBUTION_OUTPUTS:
                prediction = np.concatenate(predictions[name])
                metrics[name] = {
                    "rmse": masked_rmse(target, prediction, mask),
                    "global_r2": masked_global_r2(target, prediction, mask),
                    "median_per_protein_r2": median_per_protein_r2(target, prediction, mask),
                }
            result[scenario] = {
                "outputs": metrics,
                "valid_position_l2_norms": {
                    name: float(np.sqrt(value)) for name, value in norm_squares.items()
                },
            }
    return result
