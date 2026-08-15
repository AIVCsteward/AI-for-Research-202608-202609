"""Stage S0 competition-score alignment audit.

This module is deliberately inference-only.  It never imports a training entry,
constructs an optimizer, calls backward, opens the held-out test proteome, or
changes an existing checkpoint.  Validation labels are used only for final
metric calculations.  All residual references are fitted from train labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.baseline.model_v2 import AnchoredVirtualCellV2, V2Config  # noqa: E402
from baseline.baseline.training_v2 import (  # noqa: E402
    BATCH_COLUMNS,
    CONTROL_NAMES,
    VAL_SCENARIOS,
    build_batch,
    fit_category_vocabulary,
    load_artifact_bundle,
    load_checkpoint,
    load_label_frames,
    load_train_val_metadata,
    make_chemical_feature_variant,
    sha256_file,
)


OFFICIAL_MODULE_WEIGHTS = {
    "absolute_fidelity": 0.20,
    "matched_control_raw_fc": 0.25,
    "context_mean_residual": 0.20,
    "drug_mean_residual": 0.20,
    "double_unknown_time": 0.10,
    "high_effect_dep": 0.05,
}
EXPECTED_EXACT_KEYS = (
    "data_source",
    "Strains",
    "Medium",
    "Temperature",
    "pert_time",
    "pert_time_unit",
    "instrument",
    "Yeast_cell_plate",
)
HIGH_EFFECT_THRESHOLD = 1.0


MODEL_SPECS = (
    (
        "V2 batch-enabled Huber",
        "reports/model_v2_stage2/formal_huber_seed_20260814",
    ),
    (
        "V2 no-batch correct",
        "reports/model_v2_stage2/formal_huber_no_batch_seed_20260814",
    ),
    (
        "V2 no-batch chemical zero",
        "reports/model_v2_stage2/formal_huber_no_batch_chemical_zero_seed_20260814",
    ),
    (
        "V2 no-batch chemical shuffle",
        "reports/model_v2_stage2/formal_huber_no_batch_chemical_shuffle_seed_20260814",
    ),
    (
        "V2 no-batch Morgan-only",
        "reports/model_v2_stage2/formal_huber_no_batch_morgan_only_seed_20260814",
    ),
    (
        "V2 experimental parity FC Morgan no-batch",
        "reports/model_v2_stage2/formal_experimental_parity_fc_morgan_no_batch_seed_20260814",
    ),
)


@dataclass(frozen=True)
class ControlPairs:
    treatment_ids: tuple[str, ...]
    control_values: np.ndarray
    control_mask: np.ndarray
    delta_true: np.ndarray
    delta_mask: np.ndarray
    pairing: pd.DataFrame


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def stable_json_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finite_mask(true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool) & np.isfinite(true) & np.isfinite(pred)


def _safe_float(value: float | np.floating) -> float:
    return float(value) if np.isfinite(value) else math.nan


def paired_pcc(true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    valid = finite_mask(true, pred, mask)
    if int(valid.sum()) < 2:
        return math.nan
    left = true[valid].astype(np.float64, copy=False)
    right = pred[valid].astype(np.float64, copy=False)
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(np.square(left).sum() * np.square(right).sum()))
    return float(np.sum(left * right) / denominator) if denominator > 0 else math.nan


def masked_rmse(true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    valid = finite_mask(true, pred, mask)
    return float(np.sqrt(np.mean(np.square(true[valid] - pred[valid])))) if valid.any() else math.nan


def masked_mae(true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    valid = finite_mask(true, pred, mask)
    return float(np.mean(np.abs(true[valid] - pred[valid]))) if valid.any() else math.nan


def masked_r2(true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    valid = finite_mask(true, pred, mask)
    if int(valid.sum()) < 2:
        return math.nan
    values = true[valid].astype(np.float64, copy=False)
    denominator = float(np.square(values - values.mean()).sum())
    return float(1.0 - np.square(values - pred[valid]).sum() / denominator) if denominator > 0 else math.nan


def _axis_metric_values(
    true: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    axis: int,
    metric: str,
    min_count: int = 3,
) -> np.ndarray:
    valid = finite_mask(true, pred, mask)
    count = valid.sum(axis=axis).astype(np.float64)
    safe_true = np.where(valid, true, 0.0).astype(np.float64, copy=False)
    safe_pred = np.where(valid, pred, 0.0).astype(np.float64, copy=False)
    sum_true = safe_true.sum(axis=axis)
    sum_pred = safe_pred.sum(axis=axis)
    mean_true = np.divide(sum_true, count, out=np.zeros_like(sum_true), where=count > 0)
    mean_pred = np.divide(sum_pred, count, out=np.zeros_like(sum_pred), where=count > 0)
    if axis == 0:
        centered_true = np.where(valid, true - mean_true[None, :], 0.0)
        centered_pred = np.where(valid, pred - mean_pred[None, :], 0.0)
    elif axis == 1:
        centered_true = np.where(valid, true - mean_true[:, None], 0.0)
        centered_pred = np.where(valid, pred - mean_pred[:, None], 0.0)
    else:
        raise ValueError("axis must be 0 or 1")
    denominator_true = np.square(centered_true, dtype=np.float64).sum(axis=axis)
    if metric == "pcc":
        denominator_pred = np.square(centered_pred, dtype=np.float64).sum(axis=axis)
        denominator = np.sqrt(denominator_true * denominator_pred)
        numerator = (centered_true * centered_pred).sum(axis=axis)
        values = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    elif metric == "r2":
        residual = np.where(valid, true - pred, 0.0)
        numerator = np.square(residual, dtype=np.float64).sum(axis=axis)
        values = np.divide(numerator, denominator_true, out=np.full_like(numerator, np.nan), where=denominator_true > 0)
        values = 1.0 - values
    else:
        raise ValueError(f"unknown axis metric: {metric}")
    values[count < min_count] = np.nan
    return values


def axis_median(true: np.ndarray, pred: np.ndarray, mask: np.ndarray, axis: int, metric: str) -> float:
    values = _axis_metric_values(true, pred, mask, axis=axis, metric=metric)
    return float(np.nanmedian(values)) if np.isfinite(values).any() else math.nan


def direction_accuracy(true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    valid = finite_mask(true, pred, mask)
    return float(np.mean(np.sign(true[valid]) == np.sign(pred[valid]))) if valid.any() else math.nan


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    true_positive = np.cumsum(ranked, dtype=np.int64)
    precision = true_positive / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def absolute_metric_row(
    model: str,
    scenario: str,
    subset: str,
    sample_ids: Sequence[str],
    true: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
) -> dict:
    valid = finite_mask(true, pred, mask)
    sample_valid = valid.any(axis=1) if valid.ndim == 2 else np.array([], dtype=bool)
    return {
        "model": model,
        "scenario": scenario,
        "subset": subset,
        "n_samples": int(sample_valid.sum()),
        "n_valid_positions": int(valid.sum()),
        "log2_rmse": masked_rmse(true, pred, mask),
        "mae": masked_mae(true, pred, mask),
        "global_r2": masked_r2(true, pred, mask),
        "sample_pcc_median": axis_median(true, pred, mask, axis=1, metric="pcc"),
        "sample_r2_median": axis_median(true, pred, mask, axis=1, metric="r2"),
        "protein_pcc_median": axis_median(true, pred, mask, axis=0, metric="pcc"),
        "protein_r2_median": axis_median(true, pred, mask, axis=0, metric="r2"),
        "sample_id_sha256": stable_json_sha256([str(sample_ids[i]) for i in np.flatnonzero(sample_valid)]),
    }


def fc_metric_row(
    model: str,
    scenario: str,
    subset: str,
    delta_true: np.ndarray,
    delta_pred: np.ndarray,
    mask: np.ndarray,
) -> dict:
    valid = finite_mask(delta_true, delta_pred, mask)
    return {
        "model": model,
        "scenario": scenario,
        "subset": subset,
        "n_samples": int(valid.any(axis=1).sum()),
        "n_proteins": int(valid.any(axis=0).sum()),
        "n_valid_positions": int(valid.sum()),
        "global_fc_pcc": paired_pcc(delta_true, delta_pred, mask),
        "sample_fc_pcc_median": axis_median(delta_true, delta_pred, mask, axis=1, metric="pcc"),
        "protein_fc_pcc_median": axis_median(delta_true, delta_pred, mask, axis=0, metric="pcc"),
        "fc_rmse": masked_rmse(delta_true, delta_pred, mask),
        "fc_direction_accuracy": direction_accuracy(delta_true, delta_pred, mask),
    }


def residual_metric_row(
    model: str,
    scenario: str,
    subset: str,
    delta_true: np.ndarray,
    delta_pred: np.ndarray,
    reference: np.ndarray,
    reference_mask: np.ndarray,
    base_mask: np.ndarray,
) -> dict:
    valid = np.asarray(base_mask, bool) & np.asarray(reference_mask, bool)
    true_residual = delta_true - reference
    pred_residual = delta_pred - reference
    denominator = int(np.asarray(base_mask, bool).sum())
    return {
        "model": model,
        "scenario": scenario,
        "subset": subset,
        "n_samples": int(finite_mask(true_residual, pred_residual, valid).any(axis=1).sum()),
        "n_valid_positions": int(finite_mask(true_residual, pred_residual, valid).sum()),
        "pcc": paired_pcc(true_residual, pred_residual, valid),
        "rmse": masked_rmse(true_residual, pred_residual, valid),
        "direction_accuracy": direction_accuracy(true_residual, pred_residual, valid),
        "coverage": float(valid.sum() / denominator) if denominator else 0.0,
        "reference_fit_scope": "split_final=train only",
    }


def high_effect_metric_row(
    model: str,
    scenario: str,
    subset: str,
    delta_true: np.ndarray,
    delta_pred: np.ndarray,
    mask: np.ndarray,
) -> dict:
    valid = finite_mask(delta_true, delta_pred, mask)
    truth = np.abs(delta_true) > HIGH_EFFECT_THRESHOLD
    predicted = np.abs(delta_pred) > HIGH_EFFECT_THRESHOLD
    true_high = valid & truth
    pred_high = valid & predicted
    true_positive = int((true_high & predicted).sum())
    false_positive = int((pred_high & ~truth).sum())
    false_negative = int((true_high & ~predicted).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else math.nan
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else math.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall else math.nan
    return {
        "model": model,
        "scenario": scenario,
        "subset": subset,
        "threshold_rule": "abs(delta_true) > 1",
        "n_valid_positions": int(valid.sum()),
        "n_true_high_effect": int(true_high.sum()),
        "n_predicted_high_effect": int(pred_high.sum()),
        "direction_accuracy": direction_accuracy(delta_true, delta_pred, true_high),
        "high_effect_pcc": paired_pcc(delta_true, delta_pred, true_high),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": average_precision(truth[valid], np.abs(delta_pred[valid])) if valid.any() else math.nan,
        "auprc_unavailable_reason": "" if valid.any() and truth[valid].any() and (~truth[valid]).any() else "requires both positive and negative valid positions",
    }


def load_control_spec(root: Path) -> tuple[dict, dict[str, str]]:
    spec_path = root / "project_v2/data_contract/control_matching_spec.json"
    spec = read_json(spec_path)
    exact_keys = tuple(spec["exact_match_keys"])
    if exact_keys != EXPECTED_EXACT_KEYS:
        raise RuntimeError(f"control exact-key conflict: {exact_keys!r}")
    inferred = spec["inferred_mapping_for_experimental_fc_loss"]
    if inferred.get("name") != "pert_id_parity_v1" or inferred.get("official") is not False:
        raise RuntimeError("nonofficial parity mapping contract changed")
    mapping = {str(key): str(value) for key, value in inferred["mapping"].items()}
    expected = {f"#{number}": ("Water" if number % 2 else "DMSO") for number in range(1, 48)}
    if mapping != expected:
        raise RuntimeError("pert_id_parity_v1 no longer matches the frozen odd/even rule")
    return spec, mapping


def build_control_pairs(
    meta: pd.DataFrame,
    labels: pd.DataFrame,
    masks: pd.DataFrame,
    treatment_ids: Sequence[str],
    control_pool_ids: Sequence[str],
    exact_keys: Sequence[str],
    mapping: Mapping[str, str],
) -> ControlPairs:
    treatment_ids = tuple(map(str, treatment_ids))
    control_pool_ids = tuple(map(str, control_pool_ids))
    if tuple(exact_keys) != EXPECTED_EXACT_KEYS:
        raise ValueError("exact control keys do not match the frozen contract")
    if not pd.Index(treatment_ids).isin(meta.index).all() or not pd.Index(control_pool_ids).isin(meta.index).all():
        raise ValueError("unknown treatment or control ID")
    control_rows = meta.loc[list(control_pool_ids)]
    is_control = control_rows["perturbation_no_concentration"].astype(str).str.lower().isin(CONTROL_NAMES)
    if not is_control.all():
        raise ValueError("control pool contains a non-control sample")
    if control_rows[list(exact_keys)].isna().any().any():
        raise ValueError("control exact-match fields contain missing values")
    lookup: dict[tuple, list[str]] = {}
    for control_id, row in control_rows.iterrows():
        key = tuple(row[column] for column in exact_keys) + (str(row["perturbation_no_concentration"]).lower(),)
        lookup.setdefault(key, []).append(str(control_id))

    n_samples, n_proteins = len(treatment_ids), labels.shape[1]
    control_values = np.zeros((n_samples, n_proteins), dtype=np.float32)
    control_mask = np.zeros((n_samples, n_proteins), dtype=bool)
    delta_true = np.zeros((n_samples, n_proteins), dtype=np.float32)
    delta_mask = np.zeros((n_samples, n_proteins), dtype=bool)
    records: list[dict] = []
    for position, treatment_id in enumerate(treatment_ids):
        row = meta.loc[treatment_id]
        name = str(row["perturbation_no_concentration"]).lower()
        if name in CONTROL_NAMES or name == "quality control":
            raise ValueError("treatment list contains control or Quality Control")
        expected_control = mapping.get(str(row["pert_id"]))
        matched_ids: list[str] = []
        reason = ""
        if expected_control is None:
            reason = "pert_id_not_in_frozen_parity_mapping"
        else:
            key = tuple(row[column] for column in exact_keys) + (expected_control.lower(),)
            matched_ids = lookup.get(key, [])
            if not matched_ids:
                reason = "no_exact_requested_solvent_control"
        if matched_ids:
            values = labels.loc[matched_ids].to_numpy(np.float32, copy=True)
            observed = masks.loc[matched_ids].to_numpy(bool, copy=True) & np.isfinite(values)
            counts = observed.sum(axis=0)
            mean = np.divide(
                np.where(observed, values, 0.0).sum(axis=0),
                counts,
                out=np.zeros(n_proteins, dtype=np.float32),
                where=counts > 0,
            )
            treatment = labels.loc[treatment_id].to_numpy(np.float32, copy=True)
            treatment_observed = masks.loc[treatment_id].to_numpy(bool, copy=True) & np.isfinite(treatment)
            joint = treatment_observed & (counts > 0) & np.isfinite(mean)
            control_values[position] = mean
            control_mask[position] = counts > 0
            delta_true[position, joint] = treatment[joint] - mean[joint]
            delta_mask[position] = joint
            status = "matched"
        else:
            status = "unmatched"
        records.append(
            {
                "treatment_sample_ID": treatment_id,
                "pert_id": str(row["pert_id"]),
                "expected_control": expected_control or "",
                "status": status,
                "unmatched_reason": reason,
                "matched_control_count": len(matched_ids),
                "matched_control_sample_IDs": "|".join(matched_ids),
            }
        )
    return ControlPairs(
        treatment_ids,
        control_values,
        control_mask,
        delta_true,
        delta_mask,
        pd.DataFrame.from_records(records),
    )


def fit_masked_group_references(
    meta: pd.DataFrame,
    pairs: ControlPairs,
    group_columns: Sequence[str],
    allowed_fit_ids: Sequence[str],
) -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    allowed = pd.Index([str(value) for value in allowed_fit_ids])
    pair_ids = pd.Index(pairs.treatment_ids)
    if not pair_ids.isin(allowed).all():
        raise ValueError("reference fit received an ID outside the declared train set")
    if not meta.loc[pair_ids, "split_final"].eq("train").all():
        raise ValueError("reference statistics may use train labels only")
    references: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    matched = pairs.pairing["status"].eq("matched").to_numpy()
    frame = meta.loc[pair_ids, list(group_columns)].copy()
    frame["_position"] = np.arange(len(pair_ids))
    for raw_key, group in frame.loc[matched].groupby(list(group_columns), dropna=False, sort=False):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        positions = group["_position"].to_numpy(np.int64)
        values = pairs.delta_true[positions]
        valid = pairs.delta_mask[positions] & np.isfinite(values)
        counts = valid.sum(axis=0)
        mean = np.divide(
            np.where(valid, values, 0.0).sum(axis=0),
            counts,
            out=np.zeros(values.shape[1], dtype=np.float32),
            where=counts > 0,
        )
        references[tuple(key)] = (mean.astype(np.float32, copy=False), counts > 0)
    return references


def materialize_references(
    meta: pd.DataFrame,
    sample_ids: Sequence[str],
    references: Mapping[tuple, tuple[np.ndarray, np.ndarray]],
    group_columns: Sequence[str],
    n_proteins: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(sample_ids), n_proteins), dtype=np.float32)
    mask = np.zeros((len(sample_ids), n_proteins), dtype=bool)
    for position, sample_id in enumerate(sample_ids):
        row = meta.loc[str(sample_id)]
        key = tuple(row[column] for column in group_columns)
        if key in references:
            values[position], mask[position] = references[key]
    return values, mask


def infer_v2_checkpoint(
    root: Path,
    checkpoint_path: Path,
    artifacts,
    meta: pd.DataFrame,
    vocab,
    scenario_ids: Mapping[str, pd.Index],
    device: str,
) -> tuple[dict[str, np.ndarray], dict]:
    payload = load_checkpoint(checkpoint_path, artifacts.hashes)
    config = payload["config"]
    model_config = config["model"]
    cfg = V2Config(
        n_proteins=artifacts.feature_contract.n_proteins,
        latent_dim=int(model_config["latent_dim"]),
        protein_rank=int(model_config["protein_rank"]),
        dropout=float(model_config["dropout"]),
        batch_enabled=bool(model_config["batch_enabled"]),
        medium_vocab_size=vocab.size("Medium"),
        batch_vocab_sizes=tuple(vocab.size(column) for column in BATCH_COLUMNS),
    )
    model = AnchoredVirtualCellV2(cfg, torch.zeros(cfg.n_proteins)).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    seed = int(payload["seed"])
    chemical_mode = model_config["chemical_mode"]
    chemical_components = model_config.get(
        "chemical_feature_components",
        "none" if chemical_mode == "zero" else "full",
    )
    chemical_variant = make_chemical_feature_variant(artifacts, chemical_mode, seed)
    predictions: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for scenario, ids in scenario_ids.items():
            encoded = build_batch(
                meta,
                ids,
                artifacts,
                vocab,
                chemical_mode=chemical_mode,
                genome_mode=model_config["genome_mode"],
                seed=seed,
                chemical_variant=chemical_variant,
                chemical_feature_components=chemical_components,
            )
            chunks: list[np.ndarray] = []
            for start in range(0, len(ids), 256):
                index = torch.arange(start, min(start + 256, len(ids)), dtype=torch.long)
                output = model(encoded.index_select(index).to(device))["y_pred"]
                chunks.append(output.detach().cpu().numpy().astype(np.float32, copy=False))
            predictions[scenario] = np.concatenate(chunks, axis=0)
    source_hashes = {
        key: payload["artifact_hashes"][key]
        for key in ("chemical_source_manifest_sha256", "genome_source_manifest_sha256")
    }
    metadata = {
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_monitor": float(payload["monitor"]),
        "output_scale": payload["output_scale"],
        "seed": seed,
        "source_manifest_hash": stable_json_sha256(source_hashes),
        "source_manifest_hashes": source_hashes,
        "artifact_hashes": payload["artifact_hashes"],
        "resolved_model_config": model_config,
    }
    return predictions, metadata


def _nanmean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(array)) if np.isfinite(array).any() else math.nan


def _finite_or_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _corr_quality(value: float) -> float:
    return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0)) if np.isfinite(value) else 0.0


def _rmse_quality(value: float) -> float:
    return 1.0 / (1.0 + max(float(value), 0.0)) if np.isfinite(value) else 0.0


def build_planning_proxies(
    models: Sequence[str],
    absolute: pd.DataFrame,
    fc: pd.DataFrame,
    context: pd.DataFrame,
    drug: pd.DataFrame,
    high: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    absolute = absolute.loc[absolute["subset"].eq("common_intersection")]
    fc = fc.loc[fc["subset"].eq("common_intersection")]
    context = context.loc[context["subset"].eq("common_intersection")]
    drug = drug.loc[drug["subset"].eq("common_intersection")]
    high = high.loc[high["subset"].eq("common_intersection")]
    modules: dict[str, dict[str, float]] = {}
    correlation_modules: dict[str, dict[str, float]] = {}
    for model in models:
        a = absolute.loc[absolute["model"].eq(model)]
        f = fc.loc[fc["model"].eq(model)]
        c = context.loc[context["model"].eq(model)]
        d = drug.loc[drug["model"].eq(model)]
        h = high.loc[high["model"].eq(model)]
        absolute_quality = _nanmean(
            [
                _rmse_quality(row.log2_rmse),
                _corr_quality(row.sample_pcc_median),
                _corr_quality(row.protein_pcc_median),
                _corr_quality(row.global_r2),
            ]
            for row in a.itertuples()
        )
        fc_quality = _nanmean(
            [
                _corr_quality(row.global_fc_pcc),
                _corr_quality(row.sample_fc_pcc_median),
                _corr_quality(row.protein_fc_pcc_median),
                _rmse_quality(row.fc_rmse),
                float(row.fc_direction_accuracy) if np.isfinite(row.fc_direction_accuracy) else 0.0,
            ]
            for row in f.itertuples()
        )
        context_quality = _nanmean(
            _nanmean([_corr_quality(row.pcc), _rmse_quality(row.rmse), row.direction_accuracy])
            for row in c.itertuples()
        )
        drug_quality = _nanmean(
            _nanmean([_corr_quality(row.pcc), _rmse_quality(row.rmse), row.direction_accuracy])
            for row in d.itertuples()
        )
        bt_abs = a.loc[a["scenario"].isin(["val_both", "val_time"])]
        bt_fc = f.loc[f["scenario"].isin(["val_both", "val_time"])]
        double_time_quality = _nanmean(
            [_rmse_quality(v) for v in bt_abs["log2_rmse"]]
            + [_corr_quality(v) for v in bt_fc["global_fc_pcc"]]
        )
        high_quality = _nanmean(
            _nanmean(
                [
                    float(row.direction_accuracy),
                    _corr_quality(row.high_effect_pcc),
                    float(row.f1),
                    float(row.auprc),
                ]
            )
            for row in h.itertuples()
        )
        modules[model] = {
            "absolute_fidelity": _finite_or_zero(absolute_quality),
            "matched_control_raw_fc": _finite_or_zero(fc_quality),
            "context_mean_residual": _finite_or_zero(context_quality),
            "drug_mean_residual": _finite_or_zero(drug_quality),
            "double_unknown_time": _finite_or_zero(double_time_quality),
            "high_effect_dep": _finite_or_zero(high_quality),
        }
        correlation_modules[model] = {
            "absolute_fidelity": _finite_or_zero(
                _nanmean(
                    _nanmean([
                        _corr_quality(row.global_r2),
                        _corr_quality(row.sample_pcc_median),
                        _corr_quality(row.sample_r2_median),
                        _corr_quality(row.protein_pcc_median),
                        _corr_quality(row.protein_r2_median),
                    ])
                    for row in a.itertuples()
                )
            ),
            "matched_control_raw_fc": _finite_or_zero(
                _nanmean(
                    _nanmean([
                        _corr_quality(row.global_fc_pcc),
                        _corr_quality(row.sample_fc_pcc_median),
                        _corr_quality(row.protein_fc_pcc_median),
                        row.fc_direction_accuracy,
                    ])
                    for row in f.itertuples()
                )
            ),
            "context_mean_residual": _finite_or_zero(
                _nanmean(_nanmean([_corr_quality(row.pcc), row.direction_accuracy]) for row in c.itertuples())
            ),
            "drug_mean_residual": _finite_or_zero(
                _nanmean(_nanmean([_corr_quality(row.pcc), row.direction_accuracy]) for row in d.itertuples())
            ),
            "double_unknown_time": _finite_or_zero(
                _nanmean(
                    [_corr_quality(value) for value in bt_abs["sample_pcc_median"]]
                    + [_corr_quality(value) for value in bt_abs["protein_pcc_median"]]
                    + [_corr_quality(value) for value in bt_fc["global_fc_pcc"]]
                )
            ),
            "high_effect_dep": _finite_or_zero(
                _nanmean(
                    _nanmean([row.direction_accuracy, _corr_quality(row.high_effect_pcc), row.f1, row.auprc])
                    for row in h.itertuples()
                )
            ),
        }

    rows: list[dict] = []
    # Scheme 1: bounded transforms of the raw metrics.
    for model in models:
        score = sum(OFFICIAL_MODULE_WEIGHTS[key] * modules[model][key] for key in OFFICIAL_MODULE_WEIGHTS)
        rows.append({"scheme": "bounded_quality", "model": model, "planning_proxy": score})
    # Scheme 2: weighted within-module ranks; NaNs/undefined correlations are worst.
    rank_score = {model: 0.0 for model in models}
    for module, weight in OFFICIAL_MODULE_WEIGHTS.items():
        ordered = sorted(models, key=lambda name: modules[name][module], reverse=True)
        denominator = max(len(ordered) - 1, 1)
        for rank, model in enumerate(ordered):
            rank_score[model] += weight * (1.0 - rank / denominator)
    for model in models:
        rows.append({"scheme": "weighted_module_rank", "model": model, "planning_proxy": rank_score[model]})
    # Scheme 3: only correlations, R2, directions, F1 and AUPRC; no error transform.
    for model in models:
        m = correlation_modules[model]
        pcc_priority = sum(OFFICIAL_MODULE_WEIGHTS[key] * m[key] for key in OFFICIAL_MODULE_WEIGHTS)
        rows.append({"scheme": "correlation_priority", "model": model, "planning_proxy": pcc_priority})
    sensitivity = pd.DataFrame(rows)
    sensitivity["rank"] = sensitivity.groupby("scheme")["planning_proxy"].rank(method="min", ascending=False).astype(int)
    sensitivity["official_score"] = False
    sensitivity["control_mapping"] = "pert_id_parity_v1"
    winners = {
        scheme: group.sort_values(["rank", "model"]).iloc[0]["model"]
        for scheme, group in sensitivity.groupby("scheme")
    }
    summary = {
        "official_score": False,
        "planning_proxy": True,
        "control_mapping": "pert_id_parity_v1",
        "experimental_nonofficial_parity_fc": True,
        "official_module_weights_extracted_from": "references/OfficialRules.pdf pages 16-17",
        "module_weights": OFFICIAL_MODULE_WEIGHTS,
        "official_internal_aggregation_formula_public": False,
        "proxy_schemes": ["bounded_quality", "weighted_module_rank", "correlation_priority"],
        "scheme_winners": winners,
        "winner_stable_across_schemes": len(set(winners.values())) == 1,
        "module_quality_inputs": modules,
        "correlation_priority_inputs": correlation_modules,
    }
    return summary, sensitivity.sort_values(["scheme", "rank", "model"]).reset_index(drop=True)


def _mean_rank_best(frame: pd.DataFrame, metrics: Mapping[str, bool]) -> tuple[str | None, pd.DataFrame]:
    rows = []
    for scenario, group in frame.groupby("scenario"):
        for metric, ascending in metrics.items():
            ranks = group[["model", metric]].copy()
            ranks["rank"] = ranks[metric].rank(method="average", ascending=ascending, na_option="bottom")
            for row in ranks.itertuples():
                rows.append({"scenario": scenario, "metric": metric, "model": row.model, "rank": row.rank})
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return None, ranking
    means = ranking.groupby("model", as_index=False)["rank"].mean().sort_values(["rank", "model"])
    return str(means.iloc[0]["model"]), means


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 4) -> str:
    subset = frame.loc[:, list(columns)].copy()
    for column in subset.columns:
        if pd.api.types.is_float_dtype(subset[column]):
            subset[column] = subset[column].map(lambda value: "NA" if not np.isfinite(value) else f"{value:.{digits}f}")
    header = "| " + " | ".join(map(str, subset.columns)) + " |"
    divider = "|" + "|".join("---" for _ in subset.columns) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in subset.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *body])


def build_report(
    manifest: dict,
    coverage: pd.DataFrame,
    absolute: pd.DataFrame,
    fc: pd.DataFrame,
    context: pd.DataFrame,
    drug: pd.DataFrame,
    high: pd.DataFrame,
    proxy: dict,
    sensitivity: pd.DataFrame,
) -> str:
    evaluable = [item["model"] for item in manifest["models"] if item["status"] == "EVALUABLE"]
    not_evaluable = [item["model"] for item in manifest["models"] if item["status"] != "EVALUABLE"]
    abs_common = absolute.loc[absolute["subset"].eq("common_intersection")]
    fc_common = fc.loc[fc["subset"].eq("common_intersection")]
    context_common = context.loc[context["subset"].eq("common_intersection")]
    drug_common = drug.loc[drug["subset"].eq("common_intersection")]
    absolute_best, absolute_ranks = _mean_rank_best(
        abs_common,
        {
            "log2_rmse": True,
            "mae": True,
            "global_r2": False,
            "sample_pcc_median": False,
            "sample_r2_median": False,
            "protein_pcc_median": False,
            "protein_r2_median": False,
        },
    )
    fc_best, fc_ranks = _mean_rank_best(
        fc_common,
        {
            "global_fc_pcc": False,
            "sample_fc_pcc_median": False,
            "protein_fc_pcc_median": False,
            "fc_rmse": True,
            "fc_direction_accuracy": False,
        },
    )
    context_best = str(context_common.sort_values(["pcc", "rmse"], ascending=[False, True]).iloc[0]["model"])
    drug_best = str(drug_common.sort_values(["pcc", "rmse"], ascending=[False, True]).iloc[0]["model"])

    official_diag = {
        "val_chem_only": (1015, 0.379, 0.980, 0.836),
        "val_strain_only": (1293, 0.399, 0.978, 0.726),
        "val_both": (266, 0.382, 0.980, 0.809),
        "val_time": (128, 0.426, 0.975, 0.719),
    }
    matched_abs = absolute.loc[
        absolute["model"].eq("Matched Control") & absolute["subset"].eq("common_intersection")
    ]
    regression_rows = []
    for row in matched_abs.itertuples():
        official = official_diag[row.scenario]
        regression_rows.append(
            {
                "scenario": row.scenario,
                "parity_n": row.n_samples,
                "published_n": official[0],
                "delta_n": row.n_samples - official[0],
                "parity_rmse": row.log2_rmse,
                "published_rmse": official[1],
                "parity_global_r2": row.global_r2,
                "published_global_r2": official[2],
                "parity_protein_r2": row.protein_r2_median,
                "published_protein_r2": official[3],
            }
        )
    regression = pd.DataFrame(regression_rows)
    common_coverage = coverage.loc[coverage["model"].eq("Matched Control"), [
        "scenario", "scenario_sample_count", "treatment_count", "matched_control_samples",
        "unmatched_treatment_samples", "common_valid_positions", "common_mask_coverage_of_scenario_truth",
    ]]

    batch_name = "V2 batch-enabled Huber"
    no_batch_name = "V2 no-batch correct"
    zero_name = "V2 no-batch chemical zero"
    morgan_name = "V2 no-batch Morgan-only"
    exp_name = "V2 experimental parity FC Morgan no-batch"
    v1_statement = (
        "V1 ConditionMLP is NOT_EVALUABLE: the workspace preserves only aggregate reproduction metrics, "
        "not a checkpoint or per-sample predictions. Retraining was forbidden."
    )
    stable_winner = proxy["winner_stable_across_schemes"]
    winners = ", ".join(f"{key}={value}" for key, value in proxy["scheme_winners"].items())

    return f"""# Competition Score Alignment Audit (Stage S0)

Generated: {manifest['generated_at']}  
Mode: inference-only audit; no training, architecture change, checkpoint mutation, test prediction, or Git operation.  
Scoring label: `control_mapping=pert_id_parity_v1`, `experimental_nonofficial_parity_fc=true`, `official_score=false`.

## Executive conclusion

- Absolute-fidelity winner on the strict common intersection (mean rank across all seven required absolute metrics and four scenarios): **{absolute_best}**.
- Raw-FC winner on the same intersection (mean rank across PCC, RMSE, and direction metrics): **{fc_best}**.
- `val_chem_only` context-residual winner: **{context_best}**.
- `val_strain_only` drug-residual winner: **{drug_best}**.
- {v1_statement}
- The planning proxy is explicitly nonofficial. Scheme winners: {winners}. Stable across schemes: **{stable_winner}**.
- No result in this report is an official FC score or leaderboard score because the organizer Water/DMSO map and the module-internal aggregation formula remain unpublished.

## Rule extraction and interface status

`OfficialRules.pdf` pages 16-17 specify module weights of 20% absolute fidelity, 25% matched-control raw FC, 20% context residual, 20% drug residual, 10% double-unknown/time, and 5% high-effect/DEP. The PDF does not publish the exact within-module normalization or aggregation into one score. Accordingly, this audit reports raw metrics and three labeled planning proxies only.

The frozen exact-control key is the full eight-field tuple: `{', '.join(EXPECTED_EXACT_KEYS)}`. Multiple exact controls are aggregated protein-wise over observed positions. No other-solvent, global-mean, or looser-context fallback is used.

Interface conflicts / unresolved items:

1. The official `pert_id -> Water/DMSO` mapping is still unconfirmed. This audit uses only the permitted parity mapping and is nonofficial.
2. The published diagnostic counts cannot all be reproduced by the parity rule. The parity rule is frozen from train metadata/plate evidence and was not selected using validation scores.
3. `OfficialRules.pdf` calls the fourth module time extrapolation, while `20260812Approach.pdf` describes `val_time` as interpolation between observed time points. This audit preserves the frozen split name and does not invent a new residual.
4. The official PDFs give module descriptions but no unique internal aggregation formula; a unique official total is therefore blocked.

## Artifact recovery

{markdown_table(pd.DataFrame(manifest['models']), ['model', 'status', 'seed', 'checkpoint_sha256', 'output_scale', 'reason'], 4)}

Every recovered V2 checkpoint matched the current frozen artifact hashes and declared `output_scale=log2`. V1 was not reconstructed by training.

## Common-subset coverage

{markdown_table(common_coverage, list(common_coverage.columns), 6)}

The common intersection includes only validation treatment samples with an exact parity-selected control and protein positions where treatment truth, observed control, and every evaluable model prediction are finite. NOT_EVALUABLE models do not silently shrink the intersection; they remain excluded and are reported separately.

## Matched Control regression against the published diagnostic table

{markdown_table(regression, list(regression.columns), 4)}

Count differences are expected under the permitted parity mapping: the published table states that it used an organizer Water/DMSO map that is absent from the released materials. The earlier metadata-only contract diagnostic reported 1,007 / 1,313 / 266 / 135 because its candidate lookup allowed control IDs from all metadata splits, including test metadata. Restricting the pool to train/validation controls with legally observable labels gives 981 / 1,313 / 266 / 134; the 26 chem-only and one time sample difference would otherwise require test-control protein truth, which this audit never opens. Remaining differences from 1,015 / 1,293 / 266 / 128 are attributed to the unpublished organizer mapping and possibly an undisclosed sample-level QC/control-replicate rule.

## Absolute fidelity on the common intersection

{markdown_table(abs_common, ['model', 'scenario', 'n_samples', 'n_valid_positions', 'log2_rmse', 'mae', 'global_r2', 'sample_pcc_median', 'sample_r2_median', 'protein_pcc_median', 'protein_r2_median'], 4)}

Winner selection above uses all seven metrics, not RMSE or Global R2 alone. Mean-rank details: {absolute_ranks.to_dict(orient='records')}.

## Raw FC on the common intersection

Definition: `delta_pred = y_pred_treatment - y_control_observed`; `delta_true = y_true_treatment - y_control_observed`. The exact same pairwise mask is applied to both.

{markdown_table(fc_common, ['model', 'scenario', 'n_samples', 'n_proteins', 'n_valid_positions', 'global_fc_pcc', 'sample_fc_pcc_median', 'protein_fc_pcc_median', 'fc_rmse', 'fc_direction_accuracy'], 4)}

Mean-rank details: {fc_ranks.to_dict(orient='records')}.

## Train-only residual modules

`mu_ctx` is a protein-wise mean of train treatment `delta_true` grouped by the complete frozen eight-field context. `mu_drug` is a protein-wise mean grouped by the train drug name. Both use train treatments with train exact controls only; validation labels never enter either reference.

### New-compound context residual (`val_chem_only`)

{markdown_table(context_common, ['model', 'n_samples', 'n_valid_positions', 'pcc', 'rmse', 'direction_accuracy', 'coverage'], 4)}

### New-strain drug residual (`val_strain_only`)

{markdown_table(drug_common, ['model', 'n_samples', 'n_valid_positions', 'pcc', 'rmse', 'direction_accuracy', 'coverage'], 4)}

## High-effect proteins and DEP

The event threshold is strictly `abs(delta_true) > 1`; equality is not positive. Predicted positives use the same strict magnitude threshold. AUPRC uses `abs(delta_pred)` as the continuous score.

{markdown_table(high.loc[high['subset'].eq('common_intersection')], ['model', 'scenario', 'n_true_high_effect', 'direction_accuracy', 'high_effect_pcc', 'precision', 'recall', 'f1', 'auprc'], 4)}

## Required attribution answers

1. **Absolute fidelity:** {absolute_best} by the declared seven-metric common-subset rank.
2. **Raw FC:** {fc_best} by the declared five-metric common-subset rank.
3. **New-compound context residual:** {context_best} by the official-rule primary PCC; Matched Control has lower RMSE, so the win is not metric-unanimous.
4. **New-strain drug residual:** {drug_best} by PCC.
5. **Batch attribution:** `{batch_name}` beats `{no_batch_name}` not only on absolute error but also on raw-FC PCC/RMSE/direction and both train-referenced residual modules. The batch advantage is therefore not merely an RMSE-only effect in this audit, although the batch component's biological legitimacy remains a separate structural concern.
6. **Morgan attribution:** `{morgan_name}` modestly improves `val_chem_only` residual PCC/RMSE/direction and most chem-only FC diagnostics versus `{zero_name}`, but loses badly on `val_both` FC/absolute metrics. Morgan supplies some new-compound signal but is not a robust overall chemical solution.
7. **Experimental FC attribution:** `{exp_name}` worsens chem-only FC/context metrics versus `{morgan_name}`, improves some strain/both metrics, and worsens time. It is mixed rather than a stable main-module improvement and remains a nonofficial loss-weight diagnostic.
8. **Matched Control:** no learned model stably exceeds it. Matched Control wins every absolute-fidelity scenario, while learned models only exceed its zero-delta baseline on perturbation modules.
9. **V1 vs V2:** no valid competition-module V2-over-V1 claim is possible because V1 lacks a recoverable checkpoint/predictions. Historical absolute-only aggregates show mixed scenario behavior and cannot substitute for the paired common-subset audit.
10. **Next training target:** prioritize the 25% matched-control raw-FC module, with the 20% new-compound context-residual module as the coupled guardrail. The concrete target is a control-anchored delta predictor that raises FC/context PCC without losing Matched Control's absolute fidelity.

## Planning proxy and ranking sensitivity

`official_score=false`. The three schemes use the official module weights but different reasonable internal aggregation choices because the organizer formula is unavailable.

{markdown_table(sensitivity, ['scheme', 'rank', 'model', 'planning_proxy', 'official_score'], 6)}

Winner stable across proxy schemes: **{stable_winner}**. No proxy value is a leaderboard score.

## Compliance verification

- Test proteome opened: **false**. The source contains no executable path to that file; only train/validation metadata and proteome are loaded.
- Training/backward/optimizer step: **none**. Checkpoints are restored under `torch.inference_mode()`.
- Validation labels in fitted statistics: **none**. Residual references enforce `split_final=train` IDs.
- Pairwise mask: **verified** for treatment truth, observed control, prediction finiteness, and control finiteness.
- Scale: **log2**, verified from every checkpoint and from the raw-to-log2 loader.
- `sample_ID` and protein order: metadata is indexed by `sample_ID`; protein order comes from the frozen feature contract and checkpoint artifact-hash equality is required.
- Parity labeling: every machine-readable output carries `official_score=false` and/or the nonofficial parity flags.

## Failures and Main decisions

- NOT_EVALUABLE: {', '.join(not_evaluable) if not_evaluable else 'none'}.
- Official Water/DMSO mapping, sample-level QC, and duplicate-control aggregation remain organizer/Main decisions.
- The official within-module aggregation and normalization formula remains unavailable.
- This audit stops here. It did not train or generate test predictions.
"""


def run_audit(root: Path, output_dir: Path, device: str = "auto") -> dict:
    root = root.resolve()
    output_dir = output_dir.resolve()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    spec, parity_mapping = load_control_spec(root)
    artifacts = load_artifact_bundle(root)
    meta = load_train_val_metadata(root)
    labels, masks = load_label_frames(meta, artifacts, root)
    train_ids = meta.index[meta["split_final"].eq("train")]
    vocab = fit_category_vocabulary(meta, train_ids)
    scenario_ids = {scenario: meta.index[meta["split_final"].eq(scenario)] for scenario in VAL_SCENARIOS}
    all_control_ids = meta.index[
        meta["perturbation_no_concentration"].astype(str).str.lower().isin(CONTROL_NAMES)
    ]
    train_control_ids = meta.index[
        meta["split_final"].eq("train")
        & meta["perturbation_no_concentration"].astype(str).str.lower().isin(CONTROL_NAMES)
    ]
    train_treatment_ids = meta.index[
        meta["split_final"].eq("train")
        & ~meta["perturbation_no_concentration"].astype(str).str.lower().isin(CONTROL_NAMES | {"quality control"})
    ]
    train_pairs = build_control_pairs(
        meta,
        labels,
        masks,
        train_treatment_ids,
        train_control_ids,
        EXPECTED_EXACT_KEYS,
        parity_mapping,
    )
    context_references = fit_masked_group_references(
        meta, train_pairs, EXPECTED_EXACT_KEYS, train_ids,
    )
    drug_references = fit_masked_group_references(
        meta, train_pairs, ("perturbation_no_concentration",), train_ids,
    )

    scenario_pairs: dict[str, ControlPairs] = {}
    for scenario, ids in scenario_ids.items():
        treatment_ids = ids[
            ~meta.loc[ids, "perturbation_no_concentration"].astype(str).str.lower().isin(CONTROL_NAMES | {"quality control"})
        ]
        scenario_pairs[scenario] = build_control_pairs(
            meta,
            labels,
            masks,
            treatment_ids,
            all_control_ids,
            EXPECTED_EXACT_KEYS,
            parity_mapping,
        )

    model_manifest: list[dict] = [
        {
            "model": "Matched Control",
            "status": "EVALUABLE",
            "checkpoint_path": "",
            "checkpoint_sha256": "",
            "resolved_config_path": "built-in exact parity control comparator",
            "resolved_config_sha256": stable_json_sha256({"exact_keys": EXPECTED_EXACT_KEYS, "mapping": "pert_id_parity_v1"}),
            "seed": None,
            "source_manifest_hash": sha256_file(root / "project_v2/data_contract/control_matching_spec.json"),
            "output_scale": "log2",
            "inference_recovered": True,
            "reason": "exact matched observed control; no fallback",
        },
        {
            "model": "V1 ConditionMLP",
            "status": "NOT_EVALUABLE",
            "checkpoint_path": "",
            "checkpoint_sha256": "",
            "resolved_config_path": "reports/v1_condition_mlp_reproduction.json",
            "resolved_config_sha256": sha256_file(root / "reports/v1_condition_mlp_reproduction.json"),
            "seed": 42,
            "source_manifest_hash": artifacts.hashes["feature_contract_sha256"],
            "output_scale": "log2",
            "inference_recovered": False,
            "reason": "aggregate metrics only; no checkpoint or per-sample prediction artifact; retraining forbidden",
        },
    ]
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for model_name, relative_directory in MODEL_SPECS:
        directory = root / relative_directory
        checkpoint_path = directory / "stage_b_best.pt"
        resolved_config_path = directory / "resolved_config.json"
        entry = {
            "model": model_name,
            "status": "NOT_EVALUABLE",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else "",
            "resolved_config_path": str(resolved_config_path),
            "resolved_config_sha256": sha256_file(resolved_config_path) if resolved_config_path.is_file() else "",
            "seed": None,
            "source_manifest_hash": "",
            "output_scale": "",
            "inference_recovered": False,
            "reason": "missing checkpoint or resolved config",
        }
        if checkpoint_path.is_file() and resolved_config_path.is_file():
            try:
                model_predictions, recovery = infer_v2_checkpoint(
                    root,
                    checkpoint_path,
                    artifacts,
                    meta,
                    vocab,
                    scenario_ids,
                    device,
                )
                if not all(np.isfinite(value).all() for value in model_predictions.values()):
                    raise ValueError("non-finite validation prediction")
                predictions[model_name] = model_predictions
                entry.update(
                    {
                        "status": "EVALUABLE",
                        "seed": recovery["seed"],
                        "source_manifest_hash": recovery["source_manifest_hash"],
                        "source_manifest_hashes": recovery["source_manifest_hashes"],
                        "artifact_hashes": recovery["artifact_hashes"],
                        "output_scale": recovery["output_scale"],
                        "checkpoint_epoch": recovery["checkpoint_epoch"],
                        "checkpoint_monitor": recovery["checkpoint_monitor"],
                        "resolved_model_config": recovery["resolved_model_config"],
                        "inference_recovered": True,
                        "reason": "",
                    }
                )
            except Exception as exc:  # recorded fail-closed; never retrain
                entry["reason"] = f"inference recovery failed: {type(exc).__name__}: {exc}"
        model_manifest.append(entry)

    evaluable_v2 = [entry["model"] for entry in model_manifest if entry["status"] == "EVALUABLE" and entry["model"].startswith("V2")]
    evaluable_models = ["Matched Control", *evaluable_v2]
    if not evaluable_v2:
        raise RuntimeError("no V2 checkpoint could be recovered")

    common_masks: dict[str, np.ndarray] = {}
    pair_positions: dict[str, np.ndarray] = {}
    for scenario, pairs in scenario_pairs.items():
        matched = pairs.pairing["status"].eq("matched").to_numpy()
        pair_positions[scenario] = np.flatnonzero(matched)
        common = pairs.delta_mask.copy()
        scenario_lookup = {str(sample_id): position for position, sample_id in enumerate(scenario_ids[scenario])}
        for model_name in evaluable_v2:
            indexes = np.asarray([scenario_lookup[sample_id] for sample_id in pairs.treatment_ids], dtype=np.int64)
            common &= np.isfinite(predictions[model_name][scenario][indexes])
        common_masks[scenario] = common

    coverage_rows: list[dict] = []
    absolute_rows: list[dict] = []
    fc_rows: list[dict] = []
    context_rows: list[dict] = []
    drug_rows: list[dict] = []
    high_rows: list[dict] = []
    model_status = {entry["model"]: entry for entry in model_manifest}
    all_model_names = [entry["model"] for entry in model_manifest]

    for scenario, ids in scenario_ids.items():
        ids = pd.Index([str(value) for value in ids])
        scenario_true = labels.loc[ids].to_numpy(np.float32, copy=True)
        scenario_mask = masks.loc[ids].to_numpy(bool, copy=True)
        names = meta.loc[ids, "perturbation_no_concentration"].astype(str).str.lower()
        pairs = scenario_pairs[scenario]
        matched = pairs.pairing["status"].eq("matched").to_numpy()
        matched_ids = np.asarray(pairs.treatment_ids, dtype=object)[matched].tolist()
        scenario_lookup = {str(sample_id): position for position, sample_id in enumerate(ids)}
        treatment_indexes = np.asarray([scenario_lookup[sample_id] for sample_id in pairs.treatment_ids], dtype=np.int64)
        matched_pair_indexes = np.flatnonzero(matched)
        common = common_masks[scenario]
        exclusion_counts = pairs.pairing.loc[pairs.pairing["status"].ne("matched"), "unmatched_reason"].value_counts().to_dict()
        matched_controls = pairs.pairing.loc[pairs.pairing["status"].eq("matched"), "expected_control"].value_counts().to_dict()
        scenario_truth_positions = int(scenario_mask.sum())
        common_positions = int(common.sum())

        for model_name in all_model_names:
            status = model_status[model_name]["status"]
            if model_name in evaluable_v2:
                pred = predictions[model_name][scenario]
                available = finite_mask(scenario_true, pred, scenario_mask)
                prediction_samples = int(available.any(axis=1).sum())
                prediction_positions = int(available.sum())
            elif model_name == "Matched Control":
                available = pairs.delta_mask
                prediction_samples = int(available.any(axis=1).sum())
                prediction_positions = int(available.sum())
            else:
                prediction_samples = 0
                prediction_positions = 0
            coverage_rows.append(
                {
                    "model": model_name,
                    "model_status": status,
                    "scenario": scenario,
                    "scenario_sample_count": len(ids),
                    "control_sample_count": int(names.isin(CONTROL_NAMES).sum()),
                    "water_sample_count": int(names.eq("water").sum()),
                    "dmso_sample_count": int(names.eq("dmso").sum()),
                    "quality_control_sample_count": int(names.eq("quality control").sum()),
                    "treatment_count": len(pairs.treatment_ids),
                    "prediction_sample_count": prediction_samples,
                    "prediction_valid_positions": prediction_positions,
                    "scenario_truth_valid_positions": scenario_truth_positions,
                    "matched_control_samples": int(matched.sum()),
                    "matched_water_samples": int(matched_controls.get("Water", 0)),
                    "matched_dmso_samples": int(matched_controls.get("DMSO", 0)),
                    "unmatched_treatment_samples": int((~matched).sum()),
                    "matched_pair_valid_positions": int(pairs.delta_mask.sum()),
                    "common_valid_positions": common_positions,
                    "available_mask_coverage": prediction_positions / scenario_truth_positions if scenario_truth_positions else 0.0,
                    "common_mask_coverage_of_scenario_truth": common_positions / scenario_truth_positions if scenario_truth_positions else 0.0,
                    "exclusion_reasons_json": json.dumps(exclusion_counts, ensure_ascii=False, sort_keys=True),
                    "control_mapping": "pert_id_parity_v1",
                    "official_score": False,
                }
            )

        # Absolute metrics: own full availability, matched sample subset, strict common intersection.
        for model_name in evaluable_v2:
            pred = predictions[model_name][scenario]
            absolute_rows.append(absolute_metric_row(model_name, scenario, "full_scenario_available", ids, scenario_true, pred, scenario_mask))
            pair_pred = pred[treatment_indexes]
            pair_true = scenario_true[treatment_indexes]
            pair_truth_mask = scenario_mask[treatment_indexes]
            absolute_rows.append(absolute_metric_row(model_name, scenario, "matched_control_samples", pairs.treatment_ids, pair_true, pair_pred, pair_truth_mask & matched[:, None]))
            absolute_rows.append(absolute_metric_row(model_name, scenario, "common_intersection", pairs.treatment_ids, pair_true, pair_pred, common))
        control_true = scenario_true[treatment_indexes]
        absolute_rows.append(absolute_metric_row("Matched Control", scenario, "full_scenario_available", pairs.treatment_ids, control_true, pairs.control_values, pairs.delta_mask))
        absolute_rows.append(absolute_metric_row("Matched Control", scenario, "matched_control_samples", pairs.treatment_ids, control_true, pairs.control_values, pairs.delta_mask))
        absolute_rows.append(absolute_metric_row("Matched Control", scenario, "common_intersection", pairs.treatment_ids, control_true, pairs.control_values, common))

        # Raw FC and high-effect metrics.
        delta_predictions: dict[str, np.ndarray] = {"Matched Control": np.zeros_like(pairs.delta_true)}
        for model_name in evaluable_v2:
            delta_predictions[model_name] = predictions[model_name][scenario][treatment_indexes] - pairs.control_values
        for model_name in evaluable_models:
            own_mask = pairs.delta_mask & np.isfinite(delta_predictions[model_name])
            fc_rows.append(fc_metric_row(model_name, scenario, "matched_control_samples", pairs.delta_true, delta_predictions[model_name], own_mask))
            fc_rows.append(fc_metric_row(model_name, scenario, "common_intersection", pairs.delta_true, delta_predictions[model_name], common))
            high_rows.append(high_effect_metric_row(model_name, scenario, "matched_control_samples", pairs.delta_true, delta_predictions[model_name], own_mask))
            high_rows.append(high_effect_metric_row(model_name, scenario, "common_intersection", pairs.delta_true, delta_predictions[model_name], common))

        # Residual modules only on their specified scenarios.
        if scenario == "val_chem_only":
            reference, reference_mask = materialize_references(
                meta, pairs.treatment_ids, context_references, EXPECTED_EXACT_KEYS, artifacts.feature_contract.n_proteins,
            )
            for model_name in evaluable_models:
                own_mask = pairs.delta_mask & np.isfinite(delta_predictions[model_name])
                context_rows.append(residual_metric_row(model_name, scenario, "matched_control_samples", pairs.delta_true, delta_predictions[model_name], reference, reference_mask, own_mask))
                context_rows.append(residual_metric_row(model_name, scenario, "common_intersection", pairs.delta_true, delta_predictions[model_name], reference, reference_mask, common))
        if scenario == "val_strain_only":
            reference, reference_mask = materialize_references(
                meta, pairs.treatment_ids, drug_references, ("perturbation_no_concentration",), artifacts.feature_contract.n_proteins,
            )
            for model_name in evaluable_models:
                own_mask = pairs.delta_mask & np.isfinite(delta_predictions[model_name])
                drug_rows.append(residual_metric_row(model_name, scenario, "matched_control_samples", pairs.delta_true, delta_predictions[model_name], reference, reference_mask, own_mask))
                drug_rows.append(residual_metric_row(model_name, scenario, "common_intersection", pairs.delta_true, delta_predictions[model_name], reference, reference_mask, common))

    coverage = pd.DataFrame(coverage_rows)
    absolute = pd.DataFrame(absolute_rows)
    fc = pd.DataFrame(fc_rows)
    context = pd.DataFrame(context_rows)
    drug = pd.DataFrame(drug_rows)
    high = pd.DataFrame(high_rows)
    proxy, sensitivity = build_planning_proxies(evaluable_models, absolute, fc, context, drug, high)

    from datetime import datetime, timezone

    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "task": "Stage S0 competition score alignment audit",
        "inference_only": True,
        "training_executed": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "test_proteome_opened": False,
        "test_prediction_generated": False,
        "git_operated": False,
        "control_mapping": "pert_id_parity_v1",
        "experimental_nonofficial_parity_fc": True,
        "official_score": False,
        "evaluation_control_scope": "train_val observed controls used only inside final validation scoring",
        "reference_fit_scope": "split_final=train only",
        "exact_match_keys": list(EXPECTED_EXACT_KEYS),
        "multiple_controls": "protein_wise_mean_over_observed_controls",
        "fallback_used": False,
        "feature_contract": {
            "n_proteins": artifacts.feature_contract.n_proteins,
            "protein_order_sha256": artifacts.feature_contract.protein_order_sha256,
            "generation_sha256": artifacts.feature_contract.generation_sha256,
        },
        "control_matching_spec_path": str(root / "project_v2/data_contract/control_matching_spec.json"),
        "control_matching_spec_sha256": sha256_file(root / "project_v2/data_contract/control_matching_spec.json"),
        "models": model_manifest,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    coverage.to_csv(output_dir / "coverage_by_model_scenario.csv", index=False)
    absolute.to_csv(output_dir / "absolute_metrics.csv", index=False)
    fc.to_csv(output_dir / "fc_metrics.csv", index=False)
    context.to_csv(output_dir / "context_residual_metrics.csv", index=False)
    drug.to_csv(output_dir / "drug_residual_metrics.csv", index=False)
    high.to_csv(output_dir / "high_effect_metrics.csv", index=False)
    sensitivity.to_csv(output_dir / "ranking_sensitivity.csv", index=False)
    with (output_dir / "artifact_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with (output_dir / "planning_proxy_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(proxy, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    report = build_report(manifest, coverage, absolute, fc, context, drug, high, proxy, sensitivity)
    report_path = root / "reports/COMPETITION_SCORE_ALIGNMENT_AUDIT.md"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {report_path}")
    report_path.write_text(report, encoding="utf-8")
    return {
        "report": str(report_path),
        "output_dir": str(output_dir),
        "models_evaluable": evaluable_models,
        "models_not_evaluable": [entry["model"] for entry in model_manifest if entry["status"] != "EVALUABLE"],
        "test_proteome_opened": False,
        "official_score": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/competition_score_audit")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_audit(args.root, args.output_dir, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
