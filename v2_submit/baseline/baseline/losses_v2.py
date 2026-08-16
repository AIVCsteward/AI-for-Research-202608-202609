"""Canonical mask-aware V2 loss implementation."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    absolute: float = 1.0
    fc_absolute: float = 0.0
    fc: float = 0.0
    dep: float = 0.0
    pathway: float = 0.0
    batch_reg: float = 1e-4
    # Relative strength inside batch_regularization.  Its effective coefficient
    # in total loss is batch_reg * batch_center_strength.
    batch_center_strength: float = 1.0
    response_magnitude: float = 0.0
    batch_source_reg: float = 0.0
    batch_instrument_reg: float = 0.0
    batch_plate_reg: float = 0.0


def _valid(pred, target, mask):
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("prediction, target and mask must have identical shapes")
    return mask.bool() & torch.isfinite(pred) & torch.isfinite(target)


def masked_huber(pred, target, mask, delta: float = 1.0):
    valid = _valid(pred, target, mask)
    if not valid.any():
        return pred.sum() * 0.0
    return F.huber_loss(pred[valid], target[valid], delta=delta, reduction="mean")


def masked_huber_sum_count(pred, target, mask, delta: float = 1.0):
    """Return an exact valid-position Huber sum and count for aggregation."""
    valid = _valid(pred, target, mask)
    count = int(valid.sum())
    if count == 0:
        return pred.sum().double() * 0.0, 0
    error = (pred[valid].double() - target[valid].double()).abs()
    delta64 = torch.as_tensor(delta, dtype=torch.float64, device=error.device)
    values = torch.where(
        error <= delta64,
        0.5 * error.square(),
        delta64 * (error - 0.5 * delta64),
    )
    return values.sum(), count


def masked_mse(pred, target, mask):
    valid = _valid(pred, target, mask)
    if not valid.any():
        return pred.sum() * 0.0
    return (pred[valid] - target[valid]).square().mean()


def paired_fold_change(treatment, control, treatment_mask, control_mask):
    """Return treatment-control FC with the strict joint observation mask."""
    if treatment.shape != control.shape:
        raise ValueError("treatment and control shapes differ")
    joint = (
        treatment_mask.bool() & control_mask.bool()
        & torch.isfinite(treatment) & torch.isfinite(control)
    )
    fc = torch.where(joint, treatment - control, torch.zeros_like(treatment))
    return fc, joint


def masked_pearson_loss(pred, target, mask, eps: float = 1e-8):
    valid = _valid(pred, target, mask)
    if int(valid.sum()) < 2:
        return pred.sum() * 0.0
    left, right = pred[valid], target[valid]
    left, right = left - left.mean(), right - right.mean()
    denominator = torch.sqrt(left.square().sum() * right.square().sum())
    if not torch.isfinite(denominator) or float(denominator.detach()) <= eps:
        return pred.sum() * 0.0
    return 1.0 - (left * right).sum() / denominator.clamp_min(eps)


def masked_rowwise_pearson_loss(pred, target, mask, eps: float = 1e-8, min_count: int = 2):
    """Mean Pearson loss across valid rows without flattening sample identities."""
    valid = _valid(pred, target, mask)
    counts = valid.sum(dim=1)
    safe_counts = counts.clamp_min(1).to(pred.dtype)
    pred_mean = (torch.where(valid, pred, torch.zeros_like(pred)).sum(dim=1) / safe_counts).unsqueeze(1)
    target_mean = (torch.where(valid, target, torch.zeros_like(target)).sum(dim=1) / safe_counts).unsqueeze(1)
    left = torch.where(valid, pred - pred_mean, torch.zeros_like(pred))
    right = torch.where(valid, target - target_mean, torch.zeros_like(target))
    numerator = (left * right).sum(dim=1)
    denominator = torch.sqrt(left.square().sum(dim=1) * right.square().sum(dim=1))
    eligible = (
        (counts >= int(min_count)) & torch.isfinite(denominator)
        & (denominator > float(eps))
    )
    if not bool(eligible.any()):
        return pred.sum() * 0.0
    row_losses = 1.0 - numerator[eligible] / denominator[eligible].clamp_min(eps)
    return row_losses.mean()


def batch_regularization(delta_batch, center_strength: float = 1.0):
    """L2 plus an explicit zero-mean constraint over the current sample set."""
    l2 = delta_batch.square().mean()
    center = delta_batch.mean(dim=0).square().mean()
    return l2 + float(center_strength) * center


def response_magnitude_regularization(delta_response):
    """Mean squared response amplitude; controls are structurally zero."""
    if delta_response.ndim != 2:
        raise ValueError("delta_response must be a sample by protein matrix")
    return delta_response.square().mean()


def compute_losses(outputs, y_true, obs_mask, weights=LossWeights(), fc_pred=None, fc_true=None, fc_mask=None):
    absolute = masked_huber(outputs["y_pred"], y_true, obs_mask)
    zero = outputs["y_pred"].sum() * 0.0
    fc = zero
    fc_absolute = zero
    if weights.fc != 0 and fc_pred is not None and fc_true is not None and fc_mask is not None:
        fc = masked_rowwise_pearson_loss(fc_pred, fc_true, fc_mask)
    if weights.fc_absolute != 0 and fc_pred is not None and fc_true is not None and fc_mask is not None:
        fc_absolute = masked_huber(fc_pred, fc_true, fc_mask)
    dep = zero
    pathway = zero
    batch_reg = batch_regularization(outputs["delta_batch"], weights.batch_center_strength)
    batch_source_reg = batch_regularization(outputs.get("delta_source", torch.zeros_like(outputs["delta_batch"])), weights.batch_center_strength)
    batch_instrument_reg = batch_regularization(outputs.get("delta_instrument", torch.zeros_like(outputs["delta_batch"])), weights.batch_center_strength)
    batch_plate_reg = batch_regularization(outputs.get("delta_plate", torch.zeros_like(outputs["delta_batch"])), weights.batch_center_strength)
    response_magnitude = (
        response_magnitude_regularization(outputs["delta_response"])
        if "delta_response" in outputs else zero
    )
    total = (
        weights.absolute * absolute + weights.fc_absolute * fc_absolute + weights.fc * fc
        + weights.dep * dep + weights.pathway * pathway + weights.batch_reg * batch_reg
        + weights.response_magnitude * response_magnitude
        + weights.batch_source_reg * batch_source_reg
        + weights.batch_instrument_reg * batch_instrument_reg
        + weights.batch_plate_reg * batch_plate_reg
    )
    return {
        "loss_total": total,
        "loss_absolute": absolute,
        "loss_fc": fc,
        "loss_fc_absolute": fc_absolute,
        "loss_dep": dep,
        "loss_pathway": pathway,
        "loss_batch_reg": batch_reg,
        "loss_batch_source_reg": batch_source_reg,
        "loss_batch_instrument_reg": batch_instrument_reg,
        "loss_batch_plate_reg": batch_plate_reg,
        "loss_response_magnitude": response_magnitude,
    }
