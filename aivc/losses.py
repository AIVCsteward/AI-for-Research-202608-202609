"""Person C loss functions for response-oriented AIVC training.

The functions in this module deliberately avoid deciding whether predicted
fold change comes from ``delta_drug`` or from treatment-minus-control model
predictions.  That biological definition is supplied by the caller through an
explicit ``fc_pred`` tensor.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch


def fc_pearson_loss(fc_pred, fc_true, fc_mask, eps=1e-8, min_count=2):
    """Return ``1 - Pearson r`` over valid FC entries.

    Missing entries and non-finite values are excluded.  If correlation is
    undefined (too few observations or zero variance), the function returns a
    differentiable zero so the FC term is skipped for that batch.
    """
    if fc_pred.shape != fc_true.shape or fc_pred.shape != fc_mask.shape:
        raise ValueError("fc_pred, fc_true and fc_mask must have identical shapes")

    valid = fc_mask.bool() & torch.isfinite(fc_pred) & torch.isfinite(fc_true)
    if int(valid.sum().item()) < int(min_count):
        return fc_pred.sum() * 0.0

    predicted = fc_pred[valid]
    observed = fc_true[valid]
    predicted_centered = predicted - predicted.mean()
    observed_centered = observed - observed.mean()
    denominator = torch.sqrt(
        predicted_centered.square().sum() * observed_centered.square().sum()
    )
    if not torch.isfinite(denominator) or float(denominator.detach().item()) <= eps:
        return fc_pred.sum() * 0.0

    correlation = (predicted_centered * observed_centered).sum() / denominator.clamp_min(eps)
    correlation = correlation.clamp(-1.0, 1.0)
    return 1.0 - correlation


def residual_pearson_loss(fc_pred, fc_true, fc_mask, subtract, eps=1e-8, min_count=2):
    """Return ``1 - Pearson(fc_pred - subtract, fc_true - subtract)``.

    ``subtract`` is a per-row mean vector (e.g. the context-mean μ_ctx or the
    drug-mean μ_drug) broadcastable against ``fc_pred``.  Entries where
    ``subtract`` is non-finite are excluded by the finite check inside
    :func:`fc_pearson_loss`.  Passing ``subtract=None`` reduces to the plain
    fold-change correlation.
    """
    if subtract is None:
        return fc_pearson_loss(fc_pred, fc_true, fc_mask, eps=eps, min_count=min_count)
    return fc_pearson_loss(
        fc_pred - subtract, fc_true - subtract, fc_mask, eps=eps, min_count=min_count
    )


def residual_l2_loss(
    pred_dict: Mapping[str, torch.Tensor],
    keys=("delta_drug", "delta_strain", "delta_context"),
    weights=None,
):
    """Penalize unnecessary residual magnitude without touching calibration."""
    missing = [key for key in keys if key not in pred_dict]
    if missing:
        raise KeyError(f"Missing residual outputs: {missing}")
    if not keys:
        raise ValueError("At least one residual key is required")

    if weights is None:
        weights = {key: 1.0 for key in keys}
    weighted_losses = []
    total_weight = 0.0
    for key in keys:
        weight = float(weights.get(key, 1.0))
        if weight < 0:
            raise ValueError("Residual weights must be non-negative")
        if weight == 0:
            continue
        weighted_losses.append(pred_dict[key].square().mean() * weight)
        total_weight += weight
    if total_weight == 0:
        reference = pred_dict[keys[0]]
        return reference.sum() * 0.0
    return torch.stack(weighted_losses).sum() / total_weight


def _masked_edge_correlations(values, mask, src, dst, eps, min_pairs):
    """Compute across-sample correlations for a chunk of protein edges."""
    left = values[:, src]
    right = values[:, dst]
    if mask is None:
        valid = torch.isfinite(left) & torch.isfinite(right)
    else:
        valid = mask[:, src].bool() & mask[:, dst].bool()
        valid = valid & torch.isfinite(left) & torch.isfinite(right)

    valid_float = valid.to(values.dtype)
    counts = valid_float.sum(dim=0)
    safe_counts = counts.clamp_min(1.0)
    left_mean = torch.where(valid, left, torch.zeros_like(left)).sum(dim=0) / safe_counts
    right_mean = torch.where(valid, right, torch.zeros_like(right)).sum(dim=0) / safe_counts

    left_centered = torch.where(valid, left - left_mean, torch.zeros_like(left))
    right_centered = torch.where(valid, right - right_mean, torch.zeros_like(right))
    numerator = (left_centered * right_centered).sum(dim=0)
    denominator = torch.sqrt(
        left_centered.square().sum(dim=0) * right_centered.square().sum(dim=0)
    )
    usable = (counts >= min_pairs) & torch.isfinite(denominator) & (denominator > eps)
    correlations = numerator / denominator.clamp_min(eps)
    return correlations.clamp(-1.0, 1.0), usable


def compute_target_edge_corr(
    train_values,
    edge_index,
    mask=None,
    eps=1e-8,
    min_pairs=3,
    edge_chunk_size=1024,
):
    """Compute C3 correlation targets for a B4-provided edge list.

    C does not construct or modify the graph here.  For every edge supplied by
    B, Pearson correlation is calculated only from training rows where both
    endpoint proteins are observed.  Unusable edges receive NaN and are
    skipped later by ``correlation_consistency_loss``.
    """
    if train_values.ndim != 2:
        raise ValueError("train_values must have shape (samples, proteins)")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edges)")
    if mask is not None and mask.shape != train_values.shape:
        raise ValueError("mask must match train_values shape")
    if edge_chunk_size <= 0:
        raise ValueError("edge_chunk_size must be positive")

    values = train_values.detach()
    observed = mask.detach() if mask is not None else None
    edges = edge_index.to(device=values.device, dtype=torch.long)
    if edges.numel() and (
        int(edges.min().item()) < 0 or int(edges.max().item()) >= values.shape[1]
    ):
        raise ValueError("edge_index contains an invalid protein index")

    targets = values.new_full((edges.shape[1],), float("nan"))
    with torch.no_grad():
        for start in range(0, edges.shape[1], edge_chunk_size):
            stop = min(start + edge_chunk_size, edges.shape[1])
            correlations, usable = _masked_edge_correlations(
                values,
                observed,
                edges[0, start:stop],
                edges[1, start:stop],
                eps,
                min_pairs,
            )
            target_chunk = targets[start:stop]
            target_chunk[usable] = correlations[usable]
    return targets


def correlation_consistency_loss(
    pred_values,
    edge_index,
    target_edge_corr,
    mask=None,
    eps=1e-8,
    min_pairs=3,
    edge_chunk_size=1024,
):
    """Match predicted protein correlations on selected graph edges.

    This sparse form avoids constructing a dense ``P x P`` matrix.  The graph
    and target correlations must be computed from training rows only.
    """
    if pred_values.ndim != 2:
        raise ValueError("pred_values must have shape (batch, proteins)")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edges)")
    if target_edge_corr.ndim != 1 or target_edge_corr.shape[0] != edge_index.shape[1]:
        raise ValueError("target_edge_corr must contain one value per edge")
    if mask is not None and mask.shape != pred_values.shape:
        raise ValueError("mask must match pred_values shape")
    if edge_chunk_size <= 0:
        raise ValueError("edge_chunk_size must be positive")

    edge_index = edge_index.to(device=pred_values.device, dtype=torch.long)
    target_edge_corr = target_edge_corr.to(device=pred_values.device, dtype=pred_values.dtype)
    if edge_index.numel() and (
        int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= pred_values.shape[1]
    ):
        raise ValueError("edge_index contains an invalid protein index")

    squared_error_sum = pred_values.sum() * 0.0
    usable_edge_count = 0
    for start in range(0, edge_index.shape[1], edge_chunk_size):
        stop = min(start + edge_chunk_size, edge_index.shape[1])
        src = edge_index[0, start:stop]
        dst = edge_index[1, start:stop]
        predicted_corr, usable = _masked_edge_correlations(
            pred_values, mask, src, dst, eps, min_pairs
        )
        target_corr = target_edge_corr[start:stop]
        usable = usable & torch.isfinite(target_corr)
        if usable.any():
            squared_error_sum = squared_error_sum + (
                predicted_corr[usable] - target_corr[usable]
            ).square().sum()
            usable_edge_count += int(usable.sum().item())

    if usable_edge_count == 0:
        return pred_values.sum() * 0.0
    loss = squared_error_sum / usable_edge_count
    # Guard against NaN/Inf from numerical instability with near-zero variance
    if not torch.isfinite(loss):
        return pred_values.sum() * 0.0
    return loss
