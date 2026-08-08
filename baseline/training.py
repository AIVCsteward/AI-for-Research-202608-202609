"""Person C training utilities: mask-aware multi-objective optimization."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from baseline.losses import (
    correlation_consistency_loss,
    fc_pearson_loss,
    residual_l2_loss,
)


DEFAULT_LOSS_WEIGHTS = {
    "mse": 1.0,
    "fc": 0.3,
    "l2": 0.01,
    "corr": 0.1,
}


def mask_aware_mse(pred, y_filled, mask):
    """Mean squared error over observed protein values only."""
    valid_count = mask.sum()
    if float(valid_count.detach().item()) <= 0:
        return pred.sum() * 0.0
    diff = (pred - y_filled).square()
    return (diff * mask).sum() / valid_count


def masked_per_protein_r2_median(pred, y_filled, mask, min_observations=3, eps=1e-8):
    """Vectorized median per-protein R2 used for early stopping."""
    observed = mask.bool()
    observed_float = observed.to(pred.dtype)
    counts = observed_float.sum(dim=0)
    safe_counts = counts.clamp_min(1.0)
    means = (y_filled * observed_float).sum(dim=0) / safe_counts
    residual_ss = ((pred - y_filled).square() * observed_float).sum(dim=0)
    total_ss = ((y_filled - means).square() * observed_float).sum(dim=0)
    usable = (counts >= min_observations) & torch.isfinite(total_ss) & (total_ss > eps)
    if not usable.any():
        return pred.new_tensor(float("nan"))
    r2 = 1.0 - residual_ss[usable] / total_ss[usable].clamp_min(eps)
    # torch.median selects the lower middle value for an even-sized tensor,
    # while the official NumPy-style metric averages the two middle values.
    return torch.quantile(r2, 0.5)


def prepare_training_data(
    X_all, y_log2, mask_matrix, train_mask, split_masks, val_splits, device
):
    """Prepare train and validation tensors without using test rows."""
    X_train_t = torch.tensor(
        X_all[train_mask.values], dtype=torch.float32, device=device
    )
    y_train_t = torch.tensor(
        y_log2.loc[train_mask].fillna(0).values, dtype=torch.float32, device=device
    )
    mask_train_t = torch.tensor(
        mask_matrix.loc[train_mask].values, dtype=torch.float32, device=device
    )

    val_data = {}
    for split_name in val_splits:
        m = split_masks[split_name]
        if m.sum() == 0:
            continue
        val_data[split_name] = {
            "X": torch.tensor(X_all[m.values], dtype=torch.float32, device=device),
            "y_filled": torch.tensor(
                y_log2.loc[m].fillna(0).values, dtype=torch.float32, device=device
            ),
            "mask_t": torch.tensor(
                mask_matrix.loc[m].values, dtype=torch.float32, device=device
            ),
        }

    return X_train_t, y_train_t, mask_train_t, val_data


def prepare_fold_change_index(train_sample_ids: Sequence, pairs, device="cpu"):
    """Convert C1 matched-control pairs to a padded train-row index tensor.

    Row ``i`` corresponds to training row ``i``.  Values are indices of exact
    matched controls in the same training tensor; ``-1`` denotes padding or a
    sample without an FC target.
    """
    train_sample_ids = list(train_sample_ids)
    if len(set(train_sample_ids)) != len(train_sample_ids):
        raise ValueError("train_sample_ids must be unique")
    position = {sample_id: idx for idx, sample_id in enumerate(train_sample_ids)}

    max_controls = 0
    for control_ids in pairs["control_sample_ids"]:
        max_controls = max(max_controls, len(control_ids))
    max_controls = max(max_controls, 1)
    control_index = torch.full(
        (len(train_sample_ids), max_controls), -1, dtype=torch.long, device=device
    )

    seen_treatments = set()
    for _, pair in pairs.iterrows():
        treatment_id = pair["treatment_sample_ID"]
        control_ids = tuple(pair["control_sample_ids"])
        if treatment_id in seen_treatments:
            raise ValueError(f"Duplicate treatment pair: {treatment_id}")
        seen_treatments.add(treatment_id)
        if treatment_id not in position:
            raise ValueError(f"Treatment is absent from training rows: {treatment_id}")
        missing_controls = [control_id for control_id in control_ids if control_id not in position]
        if missing_controls:
            raise ValueError(f"Controls are absent from training rows: {missing_controls}")
        row = position[treatment_id]
        control_index[row, : len(control_ids)] = torch.tensor(
            [position[control_id] for control_id in control_ids],
            dtype=torch.long,
            device=device,
        )
    return control_index


def _as_pred_dict(model_output):
    if isinstance(model_output, Mapping):
        if "y_pred" not in model_output:
            raise KeyError("Model output dictionary must contain y_pred")
        output = dict(model_output)
        output.setdefault("y_raw", output["y_pred"])
        return output
    return {"y_pred": model_output, "y_raw": model_output}


def _slice_pred_dict(pred_dict, stop):
    return {
        key: value[:stop] if torch.is_tensor(value) and value.ndim > 0 else value
        for key, value in pred_dict.items()
    }


def _forward_with_matched_controls(model, X_train_t, batch_indices, fc_control_index):
    """Run primary samples and their unique matched controls in one forward pass."""
    batch_size = batch_indices.shape[0]
    if fc_control_index is None:
        return _as_pred_dict(model(X_train_t[batch_indices])), None, None

    batch_control_index = fc_control_index[batch_indices]
    eligible = batch_control_index.ge(0).any(dim=1)
    if not eligible.any():
        return _as_pred_dict(model(X_train_t[batch_indices])), eligible, None

    eligible_controls = batch_control_index[eligible]
    valid_slots = eligible_controls.ge(0)
    unique_controls, inverse = torch.unique(
        eligible_controls[valid_slots], sorted=False, return_inverse=True
    )
    combined_indices = torch.cat([batch_indices, unique_controls])
    combined_output = _as_pred_dict(model(X_train_t[combined_indices]))
    primary_output = _slice_pred_dict(combined_output, batch_size)

    unique_control_y_raw = combined_output["y_raw"][batch_size:]
    n_eligible, n_slots = eligible_controls.shape
    control_slots = unique_control_y_raw.new_zeros(
        (n_eligible, n_slots, unique_control_y_raw.shape[1])
    )
    control_slots[valid_slots] = unique_control_y_raw[inverse]
    control_pred_mean = control_slots.sum(dim=1) / valid_slots.sum(dim=1).clamp_min(1).unsqueeze(1)
    return primary_output, eligible, control_pred_mean


def compute_multitask_batch_loss(
    model,
    X_train_t,
    y_train_t,
    mask_train_t,
    batch_indices,
    fc_control_index=None,
    edge_index=None,
    target_edge_corr=None,
    loss_weights=None,
):
    """Compute C4 losses; FC uses treatment/control ``y_raw`` difference."""
    weights = dict(DEFAULT_LOSS_WEIGHTS)
    corr_explicitly_requested = False
    if loss_weights:
        weights.update(loss_weights)
        corr_explicitly_requested = (
            "corr" in loss_weights and float(loss_weights["corr"]) != 0.0
        )

    pred_dict, fc_eligible, control_pred_mean = _forward_with_matched_controls(
        model, X_train_t, batch_indices, fc_control_index
    )
    y_batch = y_train_t[batch_indices]
    mask_batch = mask_train_t[batch_indices]

    loss_mse = mask_aware_mse(pred_dict["y_pred"], y_batch, mask_batch)

    if all(key in pred_dict for key in ("delta_drug", "delta_strain", "delta_context")):
        loss_l2 = residual_l2_loss(pred_dict)
    else:
        loss_l2 = pred_dict["y_pred"].sum() * 0.0

    loss_fc = pred_dict["y_raw"].sum() * 0.0
    if fc_control_index is not None and fc_eligible is not None and fc_eligible.any():
        eligible_primary_indices = batch_indices[fc_eligible]
        eligible_controls = fc_control_index[eligible_primary_indices]
        valid_slots = eligible_controls.ge(0)
        safe_control_indices = eligible_controls.clamp_min(0)

        control_true = y_train_t[safe_control_indices]
        control_mask = mask_train_t[safe_control_indices].bool()
        control_mask = control_mask & valid_slots.unsqueeze(-1)
        control_counts = control_mask.sum(dim=1)
        control_true_mean = (
            torch.where(control_mask, control_true, torch.zeros_like(control_true)).sum(dim=1)
            / control_counts.clamp_min(1)
        )

        treatment_true = y_train_t[eligible_primary_indices]
        treatment_mask = mask_train_t[eligible_primary_indices].bool()
        fc_true = treatment_true - control_true_mean
        fc_mask = treatment_mask & control_counts.gt(0)
        fc_pred = pred_dict["y_raw"][fc_eligible] - control_pred_mean
        loss_fc = fc_pearson_loss(fc_pred, fc_true, fc_mask)

    loss_corr = pred_dict["y_raw"].sum() * 0.0
    corr_weight = float(weights["corr"])
    if (edge_index is None) != (target_edge_corr is None):
        raise ValueError("edge_index and target_edge_corr must be provided together")
    if corr_explicitly_requested and edge_index is None:
        raise ValueError(
            "A non-zero corr loss weight requires edge_index and target_edge_corr"
        )
    if edge_index is not None and corr_weight != 0.0:
        loss_corr = correlation_consistency_loss(
            pred_dict["y_raw"], edge_index, target_edge_corr, mask=mask_batch
        )

    loss_total = (
        float(weights["mse"]) * loss_mse
        + float(weights["fc"]) * loss_fc
        + float(weights["l2"]) * loss_l2
        + float(weights["corr"]) * loss_corr
    )
    components = {
        "loss_total": loss_total,
        "loss_mse": loss_mse,
        "loss_fc": loss_fc,
        "loss_l2": loss_l2,
        "loss_corr": loss_corr,
    }
    return loss_total, components, pred_dict


def train(
    model,
    X_train_t,
    y_train_t,
    mask_train_t,
    val_data,
    epochs=100,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    device="cpu",
    verbose=True,
    fc_control_index=None,
    edge_index=None,
    target_edge_corr=None,
    loss_weights=None,
    early_stopping_patience=20,
    early_stopping_split="val_both",
):
    """Train AIVCModel with separated loss logging and val_both early stopping."""
    del device  # tensors already define the active device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    n_train = X_train_t.shape[0]
    best_monitor = -float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {
        "train_loss": [],
        "loss_total": [],
        "loss_mse": [],
        "loss_fc": [],
        "loss_l2": [],
        "loss_corr": [],
        "val_loss": {split: [] for split in val_data},
        "val_per_protein_r2": {split: [] for split in val_data},
        "monitor": [],
    }

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_train, device=X_train_t.device)
        component_sums = {key: 0.0 for key in (
            "loss_total", "loss_mse", "loss_fc", "loss_l2", "loss_corr"
        )}
        n_batches = 0

        for start in range(0, n_train, batch_size):
            batch_indices = permutation[start : start + batch_size]
            optimizer.zero_grad()
            loss, components, _ = compute_multitask_batch_loss(
                model,
                X_train_t,
                y_train_t,
                mask_train_t,
                batch_indices,
                fc_control_index=fc_control_index,
                edge_index=edge_index,
                target_edge_corr=target_edge_corr,
                loss_weights=loss_weights,
            )
            loss.backward()
            optimizer.step()
            for key in component_sums:
                component_sums[key] += float(components[key].detach().item())
            n_batches += 1

        for key, total in component_sums.items():
            history[key].append(total / max(n_batches, 1))
        history["train_loss"].append(history["loss_total"][-1])

        model.eval()
        val_losses = []
        with torch.no_grad():
            for split_name, values in val_data.items():
                pred_dict = _as_pred_dict(model(values["X"]))
                val_loss = mask_aware_mse(
                    pred_dict["y_pred"], values["y_filled"], values["mask_t"]
                )
                val_r2 = masked_per_protein_r2_median(
                    pred_dict["y_pred"], values["y_filled"], values["mask_t"]
                )
                val_loss_value = float(val_loss.item())
                val_r2_value = float(val_r2.item())
                history["val_loss"][split_name].append(val_loss_value)
                history["val_per_protein_r2"][split_name].append(val_r2_value)
                val_losses.append(val_loss_value)

        if early_stopping_split in history["val_per_protein_r2"]:
            monitor = history["val_per_protein_r2"][early_stopping_split][-1]
        else:
            monitor = -float(np.mean(val_losses))
        if not np.isfinite(monitor):
            monitor = -float(np.mean(val_losses))
        history["monitor"].append(monitor)
        scheduler.step(monitor)

        if monitor > best_monitor:
            best_monitor = monitor
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(
                f"  Epoch {epoch + 1:3d}/{epochs} | total={history['loss_total'][-1]:.4f} "
                f"mse={history['loss_mse'][-1]:.4f} fc={history['loss_fc'][-1]:.4f} "
                f"l2={history['loss_l2'][-1]:.4f} corr={history['loss_corr'][-1]:.4f} "
                f"monitor={monitor:.4f}"
            )

        if early_stopping_patience is not None and (
            epochs_without_improvement >= early_stopping_patience
        ):
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}; best monitor={best_monitor:.4f}")
            break

    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint")
    model.load_state_dict(best_state)
    if verbose:
        print(f"训练完成, best {early_stopping_split} Per-Protein R2={best_monitor:.4f}")
    return model, history
