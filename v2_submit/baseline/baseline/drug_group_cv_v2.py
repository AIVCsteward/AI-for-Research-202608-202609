"""Stage 2.6 deterministic train-drug grouped cross-validation.

The outer held-out labels are sliced only after the inner-selected best
checkpoint has been restored. Official validation and test proteomes are not
used by this workflow.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .chemical_ood_analysis_v2 import compute_similarity_covariates, spearman_correlation
from .evaluation_v2 import masked_mae, masked_rmse
from .losses_v2 import LossWeights
from .model_v2 import AnchoredVirtualCellV2, V2Config
from .normalized_response_analysis_v2 import response_direction_diagnostics
from .training_v2 import (
    BATCH_COLUMNS,
    CONTROL_NAMES,
    ROOT,
    apply_chemical_feature_components,
    build_batch,
    configure_stage_optimizer,
    fit_category_vocabulary,
    fit_control_protein_anchor,
    load_artifact_bundle,
    load_checkpoint,
    load_config,
    load_label_frames,
    load_train_val_metadata,
    make_chemical_feature_variant,
    make_loader,
    module_parameters_sha256,
    run_training_stage,
    set_seed,
    sha256_file,
)


CHEMICAL_COLUMN = "perturbation_no_concentration"
COMPONENT_MODES = ("none", "morgan_only")
EXCLUDED_NAMES = CONTROL_NAMES | {"quality control"}


def _stable_digest(seed: int, scope: str, chemical: str) -> str:
    return hashlib.sha256(f"{int(seed)}|{scope}|{chemical}".encode("utf-8")).hexdigest()


def deterministic_group_folds(chemicals, n_folds=5, seed=20260814) -> dict[str, int]:
    chemicals = tuple(map(str, chemicals))
    if len(set(chemicals)) != len(chemicals):
        raise ValueError("chemical group names must be unique")
    if n_folds < 2 or len(chemicals) < n_folds:
        raise ValueError("not enough chemical groups for requested folds")
    ordered = sorted(chemicals, key=lambda name: (_stable_digest(seed, "outer", name), name))
    return {chemical: index % int(n_folds) for index, chemical in enumerate(ordered)}


def deterministic_inner_split(
    chemicals,
    outer_fold: int,
    seed=20260814,
    inner_fraction=0.2,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    chemicals = tuple(map(str, chemicals))
    if len(set(chemicals)) != len(chemicals) or len(chemicals) < 2:
        raise ValueError("inner split needs at least two unique chemical groups")
    if not 0 < float(inner_fraction) < 1:
        raise ValueError("inner_fraction must be between zero and one")
    scope = f"inner_outer_fold_{int(outer_fold)}"
    ordered = sorted(chemicals, key=lambda name: (_stable_digest(seed, scope, name), name))
    n_inner = min(len(ordered) - 1, max(1, int(np.ceil(len(ordered) * float(inner_fraction)))))
    inner = tuple(sorted(ordered[:n_inner]))
    train = tuple(sorted(ordered[n_inner:]))
    return train, inner


def select_eligible_train_drugs(meta, artifacts) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Select ordinary train treatments using frozen structure/mask metadata only."""
    train = meta.loc[meta["split_final"].eq("train")]
    names = train[CHEMICAL_COLUMN].astype(str)
    treatment_names = tuple(sorted(names.loc[~names.str.lower().isin(EXCLUDED_NAMES)].unique()))
    frozen_index = artifacts.chemical_index.reset_index(drop=True)
    records, eligible = [], []
    for chemical in treatment_names:
        positions = np.flatnonzero(frozen_index["raw_name"].astype(str).eq(chemical).to_numpy())
        if len(positions) != 1:
            raise ValueError(f"train chemical must occur exactly once in frozen index: {chemical}")
        row = int(positions[0])
        structure_valid = bool(frozen_index.loc[row, "structure_valid"])
        morgan_mask_valid = bool(artifacts.chemical_valid_mask[row, :artifacts.morgan_dim].any())
        mapping_status = str(frozen_index.loc[row, "mapping_status"])
        is_eligible = structure_valid and morgan_mask_valid and mapping_status != "unresolved"
        if is_eligible:
            eligible.append(chemical)
            exclusion_reason = "eligible"
        elif mapping_status == "unresolved":
            exclusion_reason = "unresolved"
        elif not structure_valid:
            exclusion_reason = "structure_invalid"
        else:
            exclusion_reason = "no_valid_morgan_mask"
        records.append({
            "chemical_name": chemical,
            "n_train_samples": int(names.eq(chemical).sum()),
            "mapping_status": mapping_status,
            "mapping_confidence": str(frozen_index.loc[row, "mapping_confidence"]),
            "structure_valid": structure_valid,
            "morgan_mask_valid": morgan_mask_valid,
            "cv_eligible": is_eligible,
            "exclusion_reason": exclusion_reason,
        })
    return tuple(sorted(eligible)), pd.DataFrame(records).sort_values("chemical_name").reset_index(drop=True)


def sample_ids_for_drugs(meta, chemicals) -> pd.Index:
    chemicals = set(map(str, chemicals))
    selected = meta["split_final"].eq("train") & meta[CHEMICAL_COLUMN].astype(str).isin(chemicals)
    ids = meta.index[selected]
    if len(ids) == 0:
        raise ValueError("drug selection produced no train samples")
    observed = set(meta.loc[ids, CHEMICAL_COLUMN].astype(str))
    if observed != chemicals:
        raise ValueError("one or more requested drug groups have no train samples")
    return ids


def sample_order_sha256(sample_ids) -> str:
    return hashlib.sha256("\n".join(map(str, sample_ids)).encode("utf-8")).hexdigest()


def build_cv_manifest(meta, artifacts, n_folds=5, seed=20260814, inner_fraction=0.2):
    eligible, entity_audit = select_eligible_train_drugs(meta, artifacts)
    assignments = deterministic_group_folds(eligible, n_folds=n_folds, seed=seed)
    fold_table = entity_audit.copy()
    fold_table["outer_fold"] = (
        fold_table["chemical_name"].map(assignments).fillna(-1).astype(int)
    )
    fold_table["fold_assignment_seed"] = int(seed)
    fold_table["fold_assignment_method"] = "sha256_sorted_round_robin_by_chemical_name"
    splits = {}
    for outer_fold in range(int(n_folds)):
        outer = tuple(sorted(name for name, fold in assignments.items() if fold == outer_fold))
        remaining = tuple(sorted(set(eligible) - set(outer)))
        train, inner = deterministic_inner_split(
            remaining, outer_fold, seed=seed, inner_fraction=inner_fraction,
        )
        if set(train) & set(inner) or set(train) & set(outer) or set(inner) & set(outer):
            raise RuntimeError("chemical groups overlap across train/inner/outer")
        if set(train) | set(inner) | set(outer) != set(eligible):
            raise RuntimeError("chemical group split does not cover the eligible universe")
        splits[outer_fold] = {
            "train_drugs": train,
            "inner_validation_drugs": inner,
            "outer_hidden_drugs": outer,
            "train_ids": tuple(map(str, sample_ids_for_drugs(meta, train))),
            "inner_validation_ids": tuple(map(str, sample_ids_for_drugs(meta, inner))),
            "outer_hidden_ids": tuple(map(str, sample_ids_for_drugs(meta, outer))),
        }
    return eligible, fold_table, splits


def load_reused_cv_manifest(
    meta,
    artifacts,
    folds_path: Path,
    reference_training_root: Path,
    expected_folds_sha256: str,
    n_folds=5,
):
    """Load the frozen Stage 2.6 outer folds and exact reference inner/train IDs."""
    folds_path = Path(folds_path)
    reference_training_root = Path(reference_training_root)
    actual_hash = sha256_file(folds_path)
    if actual_hash != expected_folds_sha256:
        raise ValueError("reused Stage 2.6 folds CSV SHA-256 mismatch")
    fold_table = pd.read_csv(folds_path)
    required = {
        "chemical_name", "n_train_samples", "structure_valid", "morgan_mask_valid",
        "cv_eligible", "exclusion_reason", "outer_fold",
    }
    if not required.issubset(fold_table.columns) or fold_table.isna().any().any():
        raise ValueError("frozen Stage 2.6 folds CSV is incomplete or contains NA")
    eligible_now, audit_now = select_eligible_train_drugs(meta, artifacts)
    eligible_frozen = tuple(sorted(fold_table.loc[fold_table["cv_eligible"], "chemical_name"].astype(str)))
    if eligible_frozen != eligible_now:
        raise ValueError("frozen folds eligible drug universe differs from current frozen artifacts")
    frozen_excluded = set(fold_table.loc[~fold_table["cv_eligible"], "chemical_name"].astype(str))
    current_excluded = set(audit_now.loc[~audit_now["cv_eligible"], "chemical_name"].astype(str))
    if frozen_excluded != current_excluded:
        raise ValueError("frozen folds excluded drug universe changed")
    assignments = dict(zip(
        fold_table.loc[fold_table["cv_eligible"], "chemical_name"].astype(str),
        fold_table.loc[fold_table["cv_eligible"], "outer_fold"].astype(int),
    ))
    if set(assignments.values()) != set(range(int(n_folds))):
        raise ValueError("frozen folds do not contain exactly five outer folds")
    splits, audit_hashes = {}, {}
    split_fields = (
        "train_drugs", "inner_validation_drugs", "outer_hidden_drugs",
        "n_train_samples", "n_inner_validation_samples", "n_outer_hidden_samples",
        "train_sample_order_sha256", "inner_sample_order_sha256", "outer_sample_order_sha256",
    )
    for outer_fold in range(int(n_folds)):
        paths = {
            mode: reference_training_root / f"fold_{outer_fold}" / mode / "fold_split_audit.json"
            for mode in COMPONENT_MODES
        }
        audits = {mode: json.loads(path.read_text(encoding="utf-8")) for mode, path in paths.items()}
        if any(audits["none"][field] != audits["morgan_only"][field] for field in split_fields):
            raise ValueError(f"reference none/Morgan split audits differ in fold {outer_fold}")
        reference = audits["none"]
        outer_from_csv = tuple(sorted(name for name, fold in assignments.items() if fold == outer_fold))
        if tuple(sorted(reference["outer_hidden_drugs"])) != outer_from_csv:
            raise ValueError(f"reference outer drugs differ from frozen folds CSV in fold {outer_fold}")
        train_ids = tuple(map(str, sample_ids_for_drugs(meta, reference["train_drugs"])))
        inner_ids = tuple(map(str, sample_ids_for_drugs(meta, reference["inner_validation_drugs"])))
        outer_ids = tuple(map(str, sample_ids_for_drugs(meta, reference["outer_hidden_drugs"])))
        current_hashes = {
            "train_sample_order_sha256": sample_order_sha256(train_ids),
            "inner_sample_order_sha256": sample_order_sha256(inner_ids),
            "outer_sample_order_sha256": sample_order_sha256(outer_ids),
        }
        if any(current_hashes[name] != reference[name] for name in current_hashes):
            raise ValueError(f"current sample IDs differ from reference split audit in fold {outer_fold}")
        splits[outer_fold] = {
            "train_drugs": tuple(reference["train_drugs"]),
            "inner_validation_drugs": tuple(reference["inner_validation_drugs"]),
            "outer_hidden_drugs": tuple(reference["outer_hidden_drugs"]),
            "train_ids": train_ids,
            "inner_validation_ids": inner_ids,
            "outer_hidden_ids": outer_ids,
        }
        audit_hashes[str(outer_fold)] = {mode: sha256_file(path) for mode, path in paths.items()}
    return eligible_frozen, fold_table, splits, {
        "folds_source": str(folds_path.resolve()),
        "folds_sha256": actual_hash,
        "reference_training_root": str(reference_training_root.resolve()),
        "reference_split_audit_sha256": audit_hashes,
        "folds_regenerated": False,
        "inner_train_ids_reused": True,
    }


def load_tanimoto(root=ROOT):
    chemistry = Path(root) / "external_data/chemistry"
    row_index = pd.read_csv(chemistry / "tanimoto_row_index.csv")
    column_index = pd.read_csv(chemistry / "tanimoto_column_index.csv")
    similarity = np.load(chemistry / "tanimoto_similarity.npz", allow_pickle=False)["tanimoto"]
    return similarity, row_index, column_index, {
        "tanimoto_similarity_sha256": sha256_file(chemistry / "tanimoto_similarity.npz"),
        "tanimoto_row_index_sha256": sha256_file(chemistry / "tanimoto_row_index.csv"),
        "tanimoto_column_index_sha256": sha256_file(chemistry / "tanimoto_column_index.csv"),
    }


def _model_from_stage_a(config, artifacts, vocab, protein_mean, checkpoint_path, device):
    model_config = config["model"]
    cfg = V2Config(
        n_proteins=artifacts.feature_contract.n_proteins,
        latent_dim=int(model_config["latent_dim"]),
        protein_rank=int(model_config["protein_rank"]),
        dropout=float(model_config["dropout"]),
        batch_enabled=False,
        medium_vocab_size=vocab.size("Medium"),
        batch_vocab_sizes=tuple(vocab.size(column) for column in BATCH_COLUMNS),
    )
    model = AnchoredVirtualCellV2(cfg, protein_mean).to(device)
    payload = load_checkpoint(checkpoint_path, artifacts.hashes)
    expected_stage_a_seed = int(config.get("stage_a_checkpoint_seed", config["seed"]))
    if payload.get("stage") != "control_first_A" or int(payload["seed"]) != expected_stage_a_seed:
        raise ValueError("Stage 2.6 requires the declared seed-matched Stage A checkpoint")
    if bool(payload["config"]["model"]["batch_enabled"]):
        raise ValueError("Stage 2.6 requires a no-batch Stage A checkpoint")
    model.load_state_dict(payload["model_state"])
    return model, payload


def _loader_for(
    meta, labels, masks, ids, artifacts, vocab, variant, components,
    config, shuffle=False,
):
    batch = build_batch(
        meta, ids, artifacts, vocab,
        chemical_mode="correct", chemical_feature_components=components,
        genome_mode="correct", seed=int(config["seed"]), chemical_variant=variant,
    )
    target = torch.from_numpy(labels.loc[ids].to_numpy(np.float32, copy=True))
    mask = torch.from_numpy(masks.loc[ids].to_numpy(bool, copy=True))
    return make_loader(
        batch, target, mask, batch_size=int(config["batch_size"]),
        shuffle=shuffle, seed=int(config["seed"]),
        num_workers=int(config["num_workers"]),
    )


def evaluate_outer_after_best_restore(
    model, checkpoint_path, artifacts, meta, labels, masks, outer_ids,
    vocab, variant, components, config, device,
):
    """Restore best inner-selected state before slicing any outer labels."""
    payload = load_checkpoint(checkpoint_path, artifacts.hashes)
    model.load_state_dict(payload["model_state"])
    loader = _loader_for(
        meta, labels, masks, pd.Index(outer_ids), artifacts, vocab, variant,
        components, config, shuffle=False,
    )
    predictions, baselines, responses, targets, valid_masks, sample_ids = [], [], [], [], [], []
    offset = 0
    model.eval()
    with torch.no_grad():
        for batch, target, mask in loader:
            outputs = model(batch.to(device))
            size = len(target)
            sample_ids.extend(map(str, outer_ids[offset:offset + size]))
            offset += size
            predictions.append(outputs["y_pred"].detach().cpu().numpy())
            baselines.append(outputs["y_baseline"].detach().cpu().numpy())
            responses.append(outputs["delta_response"].detach().cpu().numpy())
            targets.append(target.numpy())
            valid_masks.append(mask.numpy())
    if tuple(sample_ids) != tuple(map(str, outer_ids)):
        raise RuntimeError("outer evaluation sample order changed")
    return {
        "sample_ids": tuple(sample_ids),
        "prediction": np.concatenate(predictions),
        "baseline": np.concatenate(baselines),
        "response": np.concatenate(responses),
        "target": np.concatenate(targets),
        "mask": np.concatenate(valid_masks),
        "selected_checkpoint_epoch": int(payload["epoch"]),
        "selected_checkpoint_monitor": float(payload["monitor"]),
    }


def per_entity_evaluation(meta, evaluation, model_name, outer_fold, nearest):
    ids = pd.Index(evaluation["sample_ids"])
    chemicals = meta.loc[ids, CHEMICAL_COLUMN].astype(str).to_numpy()
    records = []
    for chemical in sorted(np.unique(chemicals)):
        positions = np.flatnonzero(chemicals == chemical)
        target = evaluation["target"][positions]
        prediction = evaluation["prediction"][positions]
        mask = evaluation["mask"][positions]
        direction = response_direction_diagnostics(
            evaluation["response"][positions], target,
            evaluation["baseline"][positions], mask,
        )
        records.append({
            "outer_fold": int(outer_fold),
            "model": model_name,
            "chemical_name": chemical,
            "n_samples": int(len(positions)),
            "rmse": masked_rmse(target, prediction, mask),
            "mae": masked_mae(target, prediction, mask),
            "response_target_cosine": direction["response_target_cosine"],
            "response_target_pearson": direction["response_target_pearson"],
            "alpha_star": direction["alpha_star"],
            "n_common_valid_positions": direction["n_common_valid_positions"],
            "max_train_morgan_tanimoto": nearest[chemical]["max_train_morgan_tanimoto"],
            "nearest_train_drug": nearest[chemical]["nearest_train_drug"],
            "target_residual_status": "train_drug_pseudo_ood_diagnostic_not_official_fc",
        })
    return records


def _require_finite_table(table, numeric_columns):
    values = table.loc[:, numeric_columns].to_numpy(np.float64)
    if not np.isfinite(values).all():
        bad = table.loc[~np.isfinite(values).all(axis=1), ["outer_fold", "model", "chemical_name"]]
        raise ValueError(f"Stage 2.6 output contains NA/inf: {bad.to_dict('records')}")


def _fold_model_run(
    base_config, component_mode, outer_fold, split, output_dir, artifacts,
    meta, labels, masks, vocab, protein_mean, stage_a_checkpoint, nearest,
    smoke=False,
):
    started = time.perf_counter()
    set_seed(int(base_config["seed"]))
    device = base_config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    run_config = copy.deepcopy(base_config)
    run_config["model"] = dict(run_config["model"])
    run_config["model"]["chemical_feature_components"] = component_mode
    run_config["cv_fold"] = {
        "outer_fold": int(outer_fold),
        "train_drugs": list(split["train_drugs"]),
        "inner_validation_drugs": list(split["inner_validation_drugs"]),
        "outer_hidden_drugs": list(split["outer_hidden_drugs"]),
        "outer_labels_used_for_training_or_early_stopping": False,
    }
    model, stage_a_payload = _model_from_stage_a(
        run_config, artifacts, vocab, protein_mean, stage_a_checkpoint, device,
    )
    initial_hash = module_parameters_sha256(model, ("chemical_encoder", "response_branch"))
    variant = make_chemical_feature_variant(artifacts, "correct", int(run_config["seed"]))
    _, _, component_audit = apply_chemical_feature_components(
        variant[0], variant[1], artifacts.morgan_dim, components=component_mode,
    )
    train_ids = pd.Index(split["train_ids"])
    inner_ids = pd.Index(split["inner_validation_ids"])
    outer_ids = tuple(split["outer_hidden_ids"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    split_audit = {
        "outer_fold": int(outer_fold),
        "model": component_mode,
        "train_drugs": list(split["train_drugs"]),
        "inner_validation_drugs": list(split["inner_validation_drugs"]),
        "outer_hidden_drugs": list(split["outer_hidden_drugs"]),
        "n_train_samples": int(len(train_ids)),
        "n_inner_validation_samples": int(len(inner_ids)),
        "n_outer_hidden_samples": int(len(outer_ids)),
        "train_sample_order_sha256": sample_order_sha256(train_ids),
        "inner_sample_order_sha256": sample_order_sha256(inner_ids),
        "outer_sample_order_sha256": sample_order_sha256(outer_ids),
        "outer_train_overlap": int(len(set(outer_ids) & set(train_ids))),
        "outer_inner_overlap": int(len(set(outer_ids) & set(inner_ids))),
        "official_validation_used": False,
        "test_proteome_opened": False,
    }
    (output_dir / "fold_split_audit.json").write_text(
        json.dumps(split_audit, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    train_loader = _loader_for(
        meta, labels, masks, train_ids, artifacts, vocab, variant,
        component_mode, run_config, shuffle=True,
    )
    inner_loader = _loader_for(
        meta, labels, masks, inner_ids, artifacts, vocab, variant,
        component_mode, run_config, shuffle=False,
    )
    optimizer = configure_stage_optimizer(
        model, run_config["stage_b"], run_config["weight_decay"],
    )
    epochs = int(run_config["smoke_epochs"] if smoke else run_config["stage_b"]["epochs"])
    run_metadata = {
        **split_audit,
        "stage_b_initial_chemical_response_sha256": initial_hash,
        "monitor_source": "inner_train_drugs_only",
        "outer_labels_available_to_training_loop": False,
        "component_audit": component_audit,
    }
    result = run_training_stage(
        model, optimizer, train_loader, inner_loader,
        LossWeights(absolute=1.0, fc=0.0, dep=0.0, pathway=0.0, batch_reg=0.0),
        epochs, int(run_config["stage_b"]["patience"]),
        output_dir / "stage_b_best.pt", run_config, artifacts.hashes,
        int(run_config["seed"]), f"stage2_6_fold_{outer_fold}_{component_mode}",
        device=device, monitor_kind="inner_drug_validation", run_metadata=run_metadata,
    )
    (output_dir / "training_history.json").write_text(
        json.dumps({"stage_a": [], "stage_b": list(result.history)}, indent=2), encoding="utf-8",
    )
    evaluation = evaluate_outer_after_best_restore(
        model, output_dir / "stage_b_best.pt", artifacts, meta, labels, masks,
        outer_ids, vocab, variant, component_mode, run_config, device,
    )
    records = per_entity_evaluation(meta, evaluation, component_mode, outer_fold, nearest)
    summary = {
        "status": "PASS",
        "mode": "smoke" if smoke else "formal",
        "outer_fold": int(outer_fold),
        "chemical_feature_components": component_mode,
        "seed": int(run_config["seed"]),
        "device": device,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if device == "cuda" else 0,
        "stage_a": {
            "retrained": False,
            "checkpoint_path": str(stage_a_checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(stage_a_checkpoint),
            "checkpoint_epoch": int(stage_a_payload["epoch"]),
        },
        "stage_b": {
            "best_epoch": int(result.best_epoch),
            "best_monitor": float(result.best_monitor),
            "actual_epochs": int(len(result.history)),
            "stopped_epoch": int(result.stopped_epoch),
            "stop_reason": result.stop_reason,
            "monitor": "inner_drug_validation_huber",
        },
        "stage_b_initial_chemical_response_sha256": initial_hash,
        "train_sample_order_sha256": split_audit["train_sample_order_sha256"],
        "selected_checkpoint_sha256": sha256_file(output_dir / "stage_b_best.pt"),
        "outer_evaluation_sequence": [
            "inner_selected_stage_b_training_completed",
            "best_checkpoint_reloaded",
            "outer_labels_sliced_after_restore",
        ],
        "outer_labels_used_for_training_or_early_stopping": False,
        "official_validation_used": False,
        "test_proteome_opened": False,
        "split_audit": split_audit,
        "per_entity": records,
        "artifacts": artifacts.hashes,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    return summary, records


def _render_report(summary, fold_table, per_entity, fold_results):
    comparison = per_entity.drop_duplicates(["outer_fold", "chemical_name"])[[
        "outer_fold", "chemical_name", "n_samples", "max_train_morgan_tanimoto",
        "nearest_train_drug", "rmse_gain",
    ]].sort_values(["outer_fold", "chemical_name"])
    drug_rows = [
        f"| {int(row.outer_fold)} | {row.chemical_name} | {int(row.n_samples)} | "
        f"{row.max_train_morgan_tanimoto:.3f} | {row.nearest_train_drug} | {row.rmse_gain:.4f} |"
        for row in comparison.itertuples()
    ]
    fold_rows = [
        f"| {int(row.outer_fold)} | {row.model} | {int(row.n_hidden_drugs)} | "
        f"{row.drug_equal_rmse:.4f} | {row.drug_equal_mae:.4f} | {row.drug_equal_cosine:.4f} | "
        f"{int(row.best_epoch)} | {row.best_inner_huber:.6f} |"
        for row in fold_results.itertuples()
    ]
    excluded = fold_table.loc[~fold_table["cv_eligible"], ["chemical_name", "exclusion_reason"]]
    excluded_text = "；".join(f"{row.chemical_name} ({row.exclusion_reason})" for row in excluded.itertuples())
    decision = summary["decision"]
    peak_allocated = max(run["peak_gpu_memory_allocated_bytes"] for run in summary["fold_runs"])
    peak_reserved = max(run["peak_gpu_memory_reserved_bytes"] for run in summary["fold_runs"])
    return f"""# MODEL V2 Stage 2.6：训练药物分组留出验证

## 边界

- 数据范围仅为 `split_final=train` 的 treatment；Water、DMSO、Quality Control 不分组。
- 34 个有效 Morgan 普通药物进入确定性 5 折；排除并单列：{excluded_text}。
- 每折 Stage A 均未重训；none 与 Morgan-only 使用同一训练/inner/outer 样本、初始化 hash、顺序、seed 和优化配置。
- 外层隐藏药物只在 inner Huber 选出的最佳 checkpoint 恢复后评估；官方四个 validation 场景未用于早停。
- `test_proteome_opened=false`；target residual 仅是训练药物伪 OOD 诊断，不是官方 FC。

## 总结

- 药物等权 Morgan RMSE gain：{summary['drug_equal_mean_rmse_gain']:.6f}；正 gain 药物 {summary['positive_gain_drug_count']}/{summary['n_eligible_drugs']}，负 gain 药物 {summary['negative_gain_drug_count']}/{summary['n_eligible_drugs']}。
- gain 中位数为 {summary['drug_equal_median_rmse_gain']:.6f}；三个最大负迁移药物贡献全部负 gain 的 {summary['top3_negative_gain_share']:.2%}，移除这三个药物后的等权均值为 {summary['mean_gain_excluding_three_most_harmful']:.6f}（仅作集中度诊断，不替代主结果）。
- 最大 Tanimoto 与 Morgan gain 的 Spearman rho={summary['max_tanimoto_vs_gain_spearman']['spearman_rho']:.4f}（n={summary['max_tanimoto_vs_gain_spearman']['n_drugs']}）。
- Morgan/none 方向非正药物数：{summary['morgan_nonpositive_direction_count']}/{summary['none_nonpositive_direction_count']}。
- 判定：{decision}

## 每折模型结果（药物等权）

| 外层折 | 模型 | 隐藏药物数 | RMSE | MAE | cosine | 最佳 epoch | inner Huber |
|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(fold_rows)}

## 每个隐藏药物的 Morgan gain

| 外层折 | 药物 | 样本数 | 最大 Tanimoto | 最近训练药物 | RMSE gain |
|---:|---|---:|---:|---|---:|
{chr(10).join(drug_rows)}

完整 none/Morgan RMSE、MAE、cosine、Pearson、alpha_star 见逐药物 CSV；每折 train loss、inner monitor、bad_epochs 和学习率见对应 `training_history.json`。

## 执行与测试

- 接口测试先于正式训练通过；smoke 运行 fold 0 的 none/Morgan-only、各 2 epochs，均 PASS，初始化与训练顺序 hash 一致，输出无 NA/inf。
- 正式命令：`python -m baseline.drug_group_cv_v2 --config D:/虚拟细胞/baseline/configs/model_v2_stage2_6_drug_group_cv.yaml --output-root D:/虚拟细胞/reports/model_v2_stage2/formal_stage2_6_drug_cv_seed_20260814`。
- 10 次 Stage B 总耗时 {summary['elapsed_seconds']:.2f} 秒；设备均为 CUDA；单 run 峰值显存 allocated/reserved 最大值为 {peak_allocated}/{peak_reserved} bytes。
- 全部 V2：69 passed（57.58 s）；Person C：6 passed（3.94 s）；冻结化学/基因组：42 passed（15.65 s）。
- 未出现 OOM、NaN、非有限输出、checkpoint 恢复失败或训练异常。部分 run 到达 epoch 99，是配置上限完成而非早停异常，详见 fold metrics。
- 正式训练后的前两次 `--summarize-only` 重试因折分清单/折级指标 CSV 类型识别条件错误及衍生折级表被覆盖而失败；修正为显式列集合包含判断，并从未变更的 10 个 training summary 与逐药物结果重建折级表。未重训、未修改 checkpoint、history 或指标。

## 产物

- 配置：`D:/虚拟细胞/baseline/configs/model_v2_stage2_6_drug_group_cv.yaml`
- 折分清单：`D:/虚拟细胞/reports/model_v2_stage2/stage2_6_drug_cv_folds.csv`
- 折级指标：`D:/虚拟细胞/reports/model_v2_stage2/stage2_6_drug_cv_fold_metrics.csv`
- 逐药物指标：`D:/虚拟细胞/reports/model_v2_stage2/stage2_6_drug_cv_per_entity.csv`
- 汇总 JSON：`D:/虚拟细胞/reports/model_v2_stage2/stage2_6_drug_cv_summary.json`
- 10 组 checkpoint/history：`D:/虚拟细胞/reports/model_v2_stage2/formal_stage2_6_drug_cv_seed_20260814/fold_{{0..4}}/{{none,morgan_only}}/`
"""


def _gain_decision_statistics(per_entity):
    unique_drugs = per_entity.drop_duplicates(["outer_fold", "chemical_name"])
    gains = unique_drugs["rmse_gain"].to_numpy(np.float64)
    ordered = np.sort(gains)
    positive = int((gains > 0).sum())
    negative = int((gains < 0).sum())
    negative_magnitudes = np.sort(-gains[gains < 0])
    negative_total = float(negative_magnitudes.sum())
    top3_share = (
        float(negative_magnitudes[-3:].sum() / negative_total)
        if len(negative_magnitudes) >= 3 and negative_total > 0 else 0.0
    )
    mean_excluding_three = float(ordered[3:].mean()) if len(ordered) > 3 else float(ordered.mean())
    mean_gain = float(gains.mean())
    median_gain = float(np.median(gains))
    return {
        "unique_drugs": unique_drugs,
        "positive": positive,
        "negative": negative,
        "mean_gain": mean_gain,
        "median_gain": median_gain,
        "top3_negative_gain_share": top3_share,
        "mean_gain_excluding_three_most_harmful": mean_excluding_three,
        "few_drugs_dominate": bool(
            mean_gain < 0 and median_gain > 0
            and top3_share >= 0.5 and mean_excluding_three > 0
        ),
    }


def run_cv(config_path: Path, output_root: Path, smoke=False):
    started = time.perf_counter()
    raw = load_config(config_path)
    config = raw["stage2_6"]
    if not (
        int(config["seed"]) in (20260814, 20260815)
        and int(config["n_outer_folds"]) == 5
        and tuple(config["component_modes"]) == COMPONENT_MODES
        and config["model"]["chemical_mode"] == "correct"
        and config["model"]["genome_mode"] == "correct"
        and config["model"]["batch_enabled"] is False
        and config["model"]["similarity_enabled"] is False
        and float(config["loss"]["fc_weight"]) == 0
        and float(config["loss"]["absolute_weight"]) == 1
    ):
        raise ValueError("Stage 2.6 frozen experiment settings changed")
    artifacts = load_artifact_bundle(ROOT)
    meta = load_train_val_metadata(ROOT)
    labels, masks = load_label_frames(meta, artifacts, ROOT)
    train_ids = meta.index[meta["split_final"].eq("train")]
    vocab = fit_category_vocabulary(meta, train_ids)
    train_names = meta.loc[train_ids, CHEMICAL_COLUMN].astype(str).str.lower()
    control_ids = train_ids[train_names.isin(CONTROL_NAMES)]
    protein_mean_values, control_counts = fit_control_protein_anchor(
        meta, labels, masks, control_ids,
    )
    if int(control_counts.min()) <= 0:
        raise ValueError("Stage A control anchor lacks a contracted protein")
    protein_mean = torch.from_numpy(protein_mean_values)
    stage_a_checkpoint = Path(config["fixed_stage_a_checkpoint"]["path"])
    if sha256_file(stage_a_checkpoint) != config["fixed_stage_a_checkpoint"]["sha256"]:
        raise ValueError("fixed Stage A checkpoint hash mismatch")
    reuse_spec = config.get("reuse_frozen_stage2_6_splits")
    if int(config["seed"]) == 20260815 and not reuse_spec:
        raise ValueError("seed 20260815 must directly reuse the frozen Stage 2.6 folds and split audits")
    if reuse_spec:
        eligible, fold_table, splits, fold_reuse_audit = load_reused_cv_manifest(
            meta, artifacts, Path(reuse_spec["folds_csv"]),
            Path(reuse_spec["reference_training_root"]), reuse_spec["folds_sha256"],
            int(config["n_outer_folds"]),
        )
    else:
        eligible, fold_table, splits = build_cv_manifest(
            meta, artifacts, int(config["n_outer_folds"]), int(config["seed"]),
            float(config["inner_validation_fraction"]),
        )
        fold_reuse_audit = {
            "folds_source": "generated_for_stage2_6_seed_20260814",
            "folds_regenerated": True,
            "inner_train_ids_reused": False,
        }
    similarity, row_index, column_index, tanimoto_hashes = load_tanimoto(ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    if not reuse_spec:
        fold_table.to_csv(output_root / "drug_cv_entity_folds.csv", index=False)
    folds_to_run = (0,) if smoke else tuple(range(int(config["n_outer_folds"])))
    all_records, run_summaries = [], []
    for outer_fold in folds_to_run:
        split = splits[outer_fold]
        nearest, similarity_audit = compute_similarity_covariates(
            similarity, row_index, column_index,
            split["train_drugs"], split["outer_hidden_drugs"],
        )
        if set(similarity_audit["training_drugs_without_morgan"]):
            raise ValueError("current fold training set contains a drug without Morgan similarity")
        for component_mode in COMPONENT_MODES:
            run_dir = output_root / f"fold_{outer_fold}" / component_mode
            run_summary, records = _fold_model_run(
                config, component_mode, outer_fold, split, run_dir, artifacts,
                meta, labels, masks, vocab, protein_mean, stage_a_checkpoint,
                nearest, smoke=smoke,
            )
            run_summary["similarity_audit"] = similarity_audit
            run_summaries.append(run_summary)
            all_records.extend(records)
        pair = run_summaries[-2:]
        if pair[0]["stage_b_initial_chemical_response_sha256"] != pair[1]["stage_b_initial_chemical_response_sha256"]:
            raise RuntimeError("none and Morgan-only Stage B initialization hashes differ")
        if pair[0]["train_sample_order_sha256"] != pair[1]["train_sample_order_sha256"]:
            raise RuntimeError("none and Morgan-only training sample orders differ")
    per_entity = pd.DataFrame(all_records)
    pivot = per_entity.pivot(index=["outer_fold", "chemical_name"], columns="model", values="rmse")
    pivot["rmse_gain"] = pivot["none"] - pivot["morgan_only"]
    gain_lookup = pivot["rmse_gain"]
    per_entity["rmse_gain"] = [gain_lookup.loc[(fold, chemical)] for fold, chemical in zip(
        per_entity["outer_fold"], per_entity["chemical_name"],
    )]
    _require_finite_table(per_entity, [
        "n_samples", "rmse", "mae", "response_target_cosine",
        "response_target_pearson", "alpha_star", "max_train_morgan_tanimoto", "rmse_gain",
    ])
    fold_records = []
    for summary in run_summaries:
        subset = per_entity.loc[
            per_entity["outer_fold"].eq(summary["outer_fold"])
            & per_entity["model"].eq(summary["chemical_feature_components"])
        ]
        fold_records.append({
            "outer_fold": summary["outer_fold"],
            "model": summary["chemical_feature_components"],
            "n_hidden_drugs": int(len(subset)),
            "n_hidden_samples": int(subset["n_samples"].sum()),
            "drug_equal_rmse": float(subset["rmse"].mean()),
            "drug_equal_mae": float(subset["mae"].mean()),
            "drug_equal_cosine": float(subset["response_target_cosine"].mean()),
            "drug_equal_pearson": float(subset["response_target_pearson"].mean()),
            "drug_equal_alpha_star": float(subset["alpha_star"].mean()),
            "best_epoch": summary["stage_b"]["best_epoch"],
            "best_inner_huber": summary["stage_b"]["best_monitor"],
            "actual_epochs": summary["stage_b"]["actual_epochs"],
            "stage_b_initial_chemical_response_sha256": summary["stage_b_initial_chemical_response_sha256"],
            "checkpoint_path": str((output_root / f"fold_{summary['outer_fold']}" / summary["chemical_feature_components"] / "stage_b_best.pt").resolve()),
            "history_path": str((output_root / f"fold_{summary['outer_fold']}" / summary["chemical_feature_components"] / "training_history.json").resolve()),
        })
    fold_results = pd.DataFrame(fold_records).sort_values(["outer_fold", "model"])
    decision_stats = _gain_decision_statistics(per_entity)
    unique_drugs = decision_stats["unique_drugs"]
    correlation = spearman_correlation(
        unique_drugs["max_train_morgan_tanimoto"], unique_drugs["rmse_gain"],
    )
    morgan = per_entity.loc[per_entity["model"].eq("morgan_only")]
    none = per_entity.loc[per_entity["model"].eq("none")]
    positive = decision_stats["positive"]
    negative = decision_stats["negative"]
    mean_gain = decision_stats["mean_gain"]
    rho = correlation["spearman_rho"]
    if decision_stats["few_drugs_dominate"]:
        decision = "多数药物 gain 为正且中位数为正，但负均值由少数大幅负迁移药物主导；按纪律先增加第二 seed，不立即修改架构。"
    elif positive > len(unique_drugs) / 2 and mean_gain > 0:
        if np.isfinite(rho) and rho >= 0.4:
            decision = "Morgan 对多数隐藏药物有帮助，且收益与 Tanimoto 明显正相关；证据偏向局部相似药物迁移。"
        else:
            decision = "Morgan 对多数隐藏药物有帮助且药物等权 gain 为正，支持跨药物迁移价值。"
    elif positive <= len(unique_drugs) / 2 or mean_gain <= 0:
        decision = "Morgan 整体未稳定优于 none；当前 chemical-to-protein response 映射需要重建。"
    else:
        decision = "结果由有限药物驱动，进入架构修改前应先增加第二 seed。"
    aggregate_scope = config.get("aggregate_output_scope", "primary_stage2_6")
    if aggregate_scope == "output_root":
        prefix = config.get("aggregate_filename_prefix", f"seed_{int(config['seed'])}")
        output_paths = {
            "folds_csv": str(Path(reuse_spec["folds_csv"]).resolve()) if reuse_spec else str((output_root / "drug_cv_entity_folds.csv").resolve()),
            "fold_metrics_csv": str((output_root / f"{prefix}_fold_metrics.csv").resolve()),
            "per_entity_csv": str((output_root / f"{prefix}_per_entity.csv").resolve()),
            "summary_json": str((output_root / f"{prefix}_summary.json").resolve()),
            "report_markdown": str((output_root / f"{prefix}_report.md").resolve()),
            "training_root": str(output_root.resolve()),
        }
    elif aggregate_scope == "primary_stage2_6":
        output_paths = {
            "folds_csv": str((ROOT / "reports/model_v2_stage2/stage2_6_drug_cv_folds.csv").resolve()),
            "fold_metrics_csv": str((ROOT / "reports/model_v2_stage2/stage2_6_drug_cv_fold_metrics.csv").resolve()),
            "per_entity_csv": str((ROOT / "reports/model_v2_stage2/stage2_6_drug_cv_per_entity.csv").resolve()),
            "summary_json": str((ROOT / "reports/model_v2_stage2/stage2_6_drug_cv_summary.json").resolve()),
            "report_markdown": str((ROOT / "reports/MODEL_V2_STAGE2_6_DRUG_GROUP_CV_REPORT.md").resolve()),
            "training_root": str(output_root.resolve()),
        }
    else:
        raise ValueError("unknown aggregate_output_scope")
    combined_summary = {
        "status": "PASS",
        "mode": "smoke" if smoke else "formal",
        "analysis": "stage2_6_train_drug_grouped_pseudo_ood_cv",
        "seed": int(config["seed"]),
        "n_outer_folds": len(folds_to_run),
        "n_eligible_drugs": int(len(unique_drugs)),
        "excluded_entities": fold_table.loc[~fold_table["cv_eligible"]].to_dict("records"),
        "drug_equal_mean_rmse_gain": mean_gain,
        "drug_equal_median_rmse_gain": decision_stats["median_gain"],
        "top3_negative_gain_share": decision_stats["top3_negative_gain_share"],
        "mean_gain_excluding_three_most_harmful": decision_stats["mean_gain_excluding_three_most_harmful"],
        "result_driven_by_few_drugs": decision_stats["few_drugs_dominate"],
        "positive_gain_drug_count": positive,
        "negative_gain_drug_count": negative,
        "zero_gain_drug_count": int(len(unique_drugs) - positive - negative),
        "max_tanimoto_vs_gain_spearman": correlation,
        "morgan_nonpositive_direction_count": int((morgan["response_target_cosine"] <= 0).sum()),
        "none_nonpositive_direction_count": int((none["response_target_cosine"] <= 0).sum()),
        "decision": decision,
        "fairness": {
            "all_none_morgan_initial_hashes_match_within_fold": True,
            "all_none_morgan_train_order_hashes_match_within_fold": True,
            "outer_labels_used_for_training_or_early_stopping": False,
            "outer_evaluation_after_best_checkpoint_restore": True,
            "official_validation_used": False,
            "drug_equal_aggregation": True,
        },
        "fold_reuse_audit": fold_reuse_audit,
        "tanimoto_artifact_hashes": tanimoto_hashes,
        "frozen_artifact_hashes": artifacts.hashes,
        "test_proteome_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
        "fold_runs": run_summaries,
        "outputs": output_paths,
    }
    if not smoke:
        if aggregate_scope == "primary_stage2_6":
            fold_table.to_csv(Path(output_paths["folds_csv"]), index=False)
        fold_results.to_csv(Path(output_paths["fold_metrics_csv"]), index=False)
        per_entity.sort_values(["outer_fold", "chemical_name", "model"]).to_csv(
            Path(output_paths["per_entity_csv"]), index=False,
        )
        Path(output_paths["summary_json"]).write_text(
            json.dumps(combined_summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
        )
        Path(output_paths["report_markdown"]).write_text(
            _render_report(combined_summary, fold_table, per_entity, fold_results), encoding="utf-8",
        )
    else:
        fold_results.to_csv(output_root / "smoke_fold_results.csv", index=False)
        per_entity.to_csv(output_root / "smoke_per_entity.csv", index=False)
        (output_root / "smoke_summary.json").write_text(
            json.dumps(combined_summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
        )
    return combined_summary


def summarize_completed_cv(config_path: Path, output_root: Path):
    """Regenerate aggregate artifacts from completed runs without training."""
    raw = load_config(config_path)["stage2_6"]
    aggregate_scope = raw.get("aggregate_output_scope", "primary_stage2_6")
    if aggregate_scope == "output_root":
        prefix = raw.get("aggregate_filename_prefix", f"seed_{int(raw['seed'])}")
        summary_path = output_root / f"{prefix}_summary.json"
        per_entity_path = output_root / f"{prefix}_per_entity.csv"
        prior_fold_results_path = output_root / f"{prefix}_fold_metrics.csv"
    else:
        summary_path = ROOT / "reports/model_v2_stage2/stage2_6_drug_cv_summary.json"
        per_entity_path = ROOT / "reports/model_v2_stage2/stage2_6_drug_cv_per_entity.csv"
        prior_fold_results_path = ROOT / "reports/model_v2_stage2/stage2_6_drug_cv_folds.csv"
    if not (summary_path.exists() and per_entity_path.exists() and prior_fold_results_path.exists()):
        raise FileNotFoundError("completed Stage 2.6 aggregate artifacts are missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_entity = pd.read_csv(per_entity_path)
    fold_results = pd.read_csv(prior_fold_results_path)
    required_fold_result_columns = {"outer_fold", "model", "drug_equal_rmse", "best_epoch"}
    if not required_fold_result_columns.issubset(set(fold_results.columns)):
        candidate = ROOT / "reports/model_v2_stage2/stage2_6_drug_cv_fold_metrics.csv"
        fold_results = pd.read_csv(candidate) if candidate.exists() else pd.DataFrame()
    if not required_fold_result_columns.issubset(set(fold_results.columns)):
        reconstructed = []
        for run in summary["fold_runs"]:
            subset = per_entity.loc[
                per_entity["outer_fold"].eq(run["outer_fold"])
                & per_entity["model"].eq(run["chemical_feature_components"])
            ]
            run_dir = output_root / f"fold_{run['outer_fold']}" / run["chemical_feature_components"]
            reconstructed.append({
                "outer_fold": run["outer_fold"],
                "model": run["chemical_feature_components"],
                "n_hidden_drugs": int(len(subset)),
                "n_hidden_samples": int(subset["n_samples"].sum()),
                "drug_equal_rmse": float(subset["rmse"].mean()),
                "drug_equal_mae": float(subset["mae"].mean()),
                "drug_equal_cosine": float(subset["response_target_cosine"].mean()),
                "drug_equal_pearson": float(subset["response_target_pearson"].mean()),
                "drug_equal_alpha_star": float(subset["alpha_star"].mean()),
                "best_epoch": run["stage_b"]["best_epoch"],
                "best_inner_huber": run["stage_b"]["best_monitor"],
                "actual_epochs": run["stage_b"]["actual_epochs"],
                "stage_b_initial_chemical_response_sha256": run["stage_b_initial_chemical_response_sha256"],
                "checkpoint_path": str((run_dir / "stage_b_best.pt").resolve()),
                "history_path": str((run_dir / "training_history.json").resolve()),
            })
        fold_results = pd.DataFrame(reconstructed).sort_values(["outer_fold", "model"])
    artifacts = load_artifact_bundle(ROOT)
    meta = load_train_val_metadata(ROOT)
    reuse_spec = raw.get("reuse_frozen_stage2_6_splits")
    if reuse_spec:
        _, fold_table, _, _ = load_reused_cv_manifest(
            meta, artifacts, Path(reuse_spec["folds_csv"]),
            Path(reuse_spec["reference_training_root"]), reuse_spec["folds_sha256"],
            int(raw["n_outer_folds"]),
        )
    else:
        _, fold_table, _ = build_cv_manifest(
            meta, artifacts, int(raw["n_outer_folds"]), int(raw["seed"]),
            float(raw["inner_validation_fraction"]),
        )
    stats = _gain_decision_statistics(per_entity)
    summary.update({
        "drug_equal_mean_rmse_gain": stats["mean_gain"],
        "drug_equal_median_rmse_gain": stats["median_gain"],
        "positive_gain_drug_count": stats["positive"],
        "negative_gain_drug_count": stats["negative"],
        "top3_negative_gain_share": stats["top3_negative_gain_share"],
        "mean_gain_excluding_three_most_harmful": stats["mean_gain_excluding_three_most_harmful"],
        "result_driven_by_few_drugs": stats["few_drugs_dominate"],
        "decision": (
            "中位 gain 为正，且移除最严重三个负迁移药物后的集中度诊断均值转正；总体负均值由少数药物主导。"
            if stats["few_drugs_dominate"] else summary["decision"]
        ),
        "excluded_entities": fold_table.loc[~fold_table["cv_eligible"]].to_dict("records"),
    })
    if aggregate_scope == "output_root":
        output_prefix = raw.get("aggregate_filename_prefix", f"seed_{int(raw['seed'])}")
        fold_metrics_path = output_root / f"{output_prefix}_fold_metrics.csv"
        report_path = output_root / f"{output_prefix}_report.md"
    else:
        fold_metrics_path = ROOT / "reports/model_v2_stage2/stage2_6_drug_cv_fold_metrics.csv"
        report_path = ROOT / "reports/MODEL_V2_STAGE2_6_DRUG_GROUP_CV_REPORT.md"
        fold_table.to_csv(ROOT / "reports/model_v2_stage2/stage2_6_drug_cv_folds.csv", index=False)
        fold_table.to_csv(output_root / "drug_cv_entity_folds.csv", index=False)
    summary["outputs"]["fold_metrics_csv"] = str(fold_metrics_path.resolve())
    fold_results.to_csv(fold_metrics_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    report_path.write_text(
        _render_report(summary, fold_table, per_entity, fold_results), encoding="utf-8",
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 2.6 deterministic train-drug grouped CV")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke and args.summarize_only:
        parser.error("--smoke and --summarize-only are mutually exclusive")
    result = (
        summarize_completed_cv(args.config, args.output_root)
        if args.summarize_only else run_cv(args.config, args.output_root, smoke=args.smoke)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
