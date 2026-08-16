"""Stage S2A: Stage-A-only hierarchical technical-batch calibration.

Only train-control labels may affect gradients, anchors, vocabularies, inner
early stopping, and checkpoints.  Validation controls are final evaluation
only.  Test metadata is read for category-overlap counts; test proteome paths
are neither accepted nor opened.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from baseline.baseline.evaluation_v2 import (
    masked_global_r2, masked_mae, masked_rmse, median_per_protein_pcc,
    median_per_sample_pcc,
)
from baseline.baseline.losses_v2 import batch_regularization, masked_huber, masked_huber_sum_count
from baseline.baseline.model_v2 import (
    BaselineBranch, BatchBranch, GenomeEncoder, HierarchicalBatchBranch,
    V2Batch, V2Config,
)
from baseline.baseline.training_v2 import (
    BATCH_COLUMNS, CONTROL_NAMES, CategoryVocabulary, build_batch,
    fit_category_vocabulary, fit_control_protein_anchor, load_artifact_bundle,
    load_config, load_label_frames, load_train_val_metadata,
    make_chemical_feature_variant, module_parameters_sha256, set_seed, sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
STRUCTURES = ("no_batch", "flat_batch", "hierarchical_batch")
GROUP_CV = {"plate": "Yeast_cell_plate", "instrument": "instrument"}


def stable_hash(values) -> str:
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _ids_hash(ids) -> str:
    return stable_hash(list(map(str, ids)))


def _resolve_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return value


class StageABatchCalibrationModel(nn.Module):
    def __init__(self, cfg: V2Config, structure: str, anchor: torch.Tensor):
        super().__init__()
        if structure not in STRUCTURES:
            raise ValueError(f"unknown Stage-A batch structure: {structure}")
        self.cfg, self.structure = cfg, structure
        self.genome_encoder = GenomeEncoder(cfg)
        self.baseline_branch = BaselineBranch(cfg)
        if structure == "flat_batch":
            self.flat_batch = BatchBranch(cfg)
        elif structure == "hierarchical_batch":
            self.hierarchical_batch = HierarchicalBatchBranch(cfg)
        if anchor.shape != (cfg.n_proteins,):
            raise ValueError("anchor shape does not match protein contract")
        self.register_buffer("train_protein_mean", anchor.float())

    def forward(self, batch: V2Batch):
        genome = self.genome_encoder(
            batch.genome, batch.genome_valid_mask, batch.genome_mapping,
            batch.genome_confidence, batch.genome_proxy,
        )
        baseline = self.train_protein_mean + self.baseline_branch(
            genome, batch.medium, batch.condition_numeric,
        )
        zero = torch.zeros_like(baseline)
        source = instrument = plate = zero
        if self.structure == "flat_batch":
            delta_batch = self.flat_batch(batch.batch_categorical)
        elif self.structure == "hierarchical_batch":
            source, instrument, plate = self.hierarchical_batch(batch.batch_categorical)
            delta_batch = source + instrument + plate
        else:
            delta_batch = zero
        return {
            "y_pred": baseline + delta_batch,
            "y_anchor": baseline + delta_batch,
            "y_baseline": baseline,
            "delta_batch": delta_batch,
            "delta_source": source,
            "delta_instrument": instrument,
            "delta_plate": plate,
            "delta_response": zero,
        }


class StageADataset(Dataset):
    def __init__(self, batch, labels, masks):
        self.batch, self.labels, self.masks = batch, labels, masks

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return index

    def collate(self, indexes):
        index = torch.as_tensor(indexes, dtype=torch.long)
        return self.batch.index_select(index), self.labels[index], self.masks[index]


def make_loader(batch, labels, masks, batch_size, shuffle, seed, num_workers=0):
    dataset = StageADataset(batch, labels, masks)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(int(seed)), collate_fn=dataset.collate,
        num_workers=int(num_workers),
    )


def deterministic_group_folds(meta, control_ids, column, n_folds=5, seed=20260814):
    ids = pd.Index(map(str, control_ids))
    if not meta.loc[ids, "split_final"].eq("train").all():
        raise ValueError("outer CV group folds accept train controls only")
    names = meta.loc[ids, "perturbation_no_concentration"].astype(str).str.lower()
    if not names.isin(CONTROL_NAMES).all():
        raise ValueError("outer CV contains non-control sample")
    groups = meta.loc[ids].groupby(column).size().to_dict()
    if len(groups) < n_folds:
        raise ValueError(f"{column} has fewer groups than requested folds")
    fold_groups = [[] for _ in range(n_folds)]
    fold_counts = [0] * n_folds
    ordered = sorted(
        groups,
        key=lambda group: (-groups[group], hashlib.sha256(f"{seed}|{group}".encode()).hexdigest()),
    )
    for group in ordered:
        target = min(range(n_folds), key=lambda fold: (fold_counts[fold], fold))
        fold_groups[target].append(str(group))
        fold_counts[target] += int(groups[group])
    values = meta.loc[ids, column].astype(str)
    folds = []
    for fold, held_groups in enumerate(fold_groups):
        holdout = ids[values.isin(held_groups).to_numpy()]
        train = ids[~values.isin(held_groups).to_numpy()]
        if set(meta.loc[train, column].astype(str)) & set(meta.loc[holdout, column].astype(str)):
            raise RuntimeError("outer group leakage")
        folds.append({
            "fold": fold, "column": column, "train_ids": train, "holdout_ids": holdout,
            "holdout_groups": tuple(sorted(held_groups)),
        })
    return folds


def split_inner_train(ids, fraction, seed, fold, structure):
    ids = pd.Index(map(str, ids))
    ordered = sorted(ids, key=lambda value: hashlib.sha256(f"{seed}|{fold}|inner|{value}".encode()).hexdigest())
    n_inner = max(1, int(round(len(ordered) * float(fraction))))
    inner = pd.Index(ordered[:n_inner])
    fit = ids[~ids.isin(inner)]
    if len(fit) == 0:
        raise ValueError("empty inner fit partition")
    return fit, inner


def _model_config(config, artifacts, vocab, structure):
    model_cfg = config["model"]
    return V2Config(
        n_proteins=artifacts.feature_contract.n_proteins,
        latent_dim=int(model_cfg["latent_dim"]), protein_rank=int(model_cfg["protein_rank"]),
        dropout=float(model_cfg["dropout"]), batch_enabled=structure != "no_batch",
        medium_vocab_size=vocab.size("Medium"),
        batch_vocab_sizes=tuple(vocab.size(column) for column in BATCH_COLUMNS),
    )


def _batch_for(meta, ids, artifacts, vocab, chemical_variant, seed):
    # build_batch materializes frozen entity features, but this Stage-A-only
    # model has no chemical encoder and never consumes those tensors.
    return build_batch(
        meta, ids, artifacts, vocab, chemical_mode="correct", genome_mode="correct",
        seed=seed, chemical_variant=chemical_variant, chemical_feature_components="none",
    )


def _loss(model, outputs, target, mask, config):
    loss = masked_huber(outputs["y_pred"], target, mask)
    loss_cfg = config["loss"]
    if model.structure == "flat_batch":
        loss = loss + float(loss_cfg["flat_batch_reg_weight"]) * batch_regularization(
            outputs["delta_batch"], float(loss_cfg["center_strength"]),
        )
    elif model.structure == "hierarchical_batch":
        for level in ("source", "instrument", "plate"):
            loss = loss + float(loss_cfg[f"hierarchical_{level}_reg_weight"]) * batch_regularization(
                outputs[f"delta_{level}"], float(loss_cfg["center_strength"]),
            )
    return loss


def _optimizer(model, config):
    rates = config["training"]["learning_rates"]
    groups = [
        {"params": model.genome_encoder.parameters(), "lr": float(rates["genome_encoder"]), "name": "genome_encoder"},
        {"params": model.baseline_branch.parameters(), "lr": float(rates["baseline_branch"]), "name": "baseline_branch"},
    ]
    if model.structure == "flat_batch":
        groups.append({"params": model.flat_batch.parameters(), "lr": float(rates["flat_batch"]), "name": "flat_batch"})
    elif model.structure == "hierarchical_batch":
        for level in ("source", "instrument", "plate"):
            groups.append({
                "params": getattr(model.hierarchical_batch, level).parameters(),
                "lr": float(rates[f"hierarchical_{level}"]), "name": f"hierarchical_{level}",
            })
    return torch.optim.AdamW(groups, weight_decay=float(config["training"]["weight_decay"]))


def _exact_huber(model, loader, device):
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch, target, mask in loader:
            output = model(batch.to(device))["y_pred"]
            value, n = masked_huber_sum_count(output, target.to(device), mask.to(device))
            total += float(value); count += n
    if count == 0:
        raise ValueError("empty monitor mask")
    return total / count


def train_with_inner_early_stopping(model, train_loader, inner_loader, config, epochs, patience, checkpoint, metadata, device):
    optimizer = _optimizer(model, config)
    best, bad, best_epoch, history = float("inf"), 0, -1, []
    for epoch in range(int(epochs)):
        model.train(); losses = []
        for batch, target, mask in train_loader:
            batch, target, mask = batch.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            value = _loss(model, model(batch), target, mask, config)
            if not torch.isfinite(value):
                raise FloatingPointError("non-finite Stage-A loss")
            value.backward(); optimizer.step(); losses.append(float(value.detach()))
        monitor = _exact_huber(model, inner_loader, device)
        if monitor < best:
            best, bad, best_epoch = monitor, 0, epoch
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": copy.deepcopy(model.state_dict()),
                "optimizer_state": copy.deepcopy(optimizer.state_dict()),
                "epoch": epoch, "monitor": monitor, "early_stopping": {"bad_epochs": 0, "patience": int(patience)},
                **metadata,
            }, checkpoint)
        else:
            bad += 1
        history.append({
            "epoch": epoch, "train_loss": float(np.mean(losses)), "monitor": monitor,
            "bad_epochs": bad, "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
        })
        if bad >= int(patience):
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    return history, payload, ("early_stopping_patience_exhausted" if bad >= int(patience) else "configured_max_epochs_completed")


def train_fixed_epochs(model, loader, config, epochs, checkpoint, metadata, device):
    optimizer = _optimizer(model, config); history = []
    for epoch in range(int(epochs)):
        model.train(); losses = []
        for batch, target, mask in loader:
            batch, target, mask = batch.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            value = _loss(model, model(batch), target, mask, config)
            value.backward(); optimizer.step(); losses.append(float(value.detach()))
        history.append({
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "monitor": None, "bad_epochs": 0,
            "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
        })
    payload = {
        "model_state": copy.deepcopy(model.state_dict()), "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "epoch": int(epochs) - 1, "monitor": None, "early_stopping": None, **metadata,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, checkpoint)
    return history, payload


def predict(model, loader, device):
    outputs = {key: [] for key in ("y_pred", "delta_batch", "delta_source", "delta_instrument", "delta_plate")}
    targets, masks = [], []
    model.eval()
    with torch.no_grad():
        for batch, target, mask in loader:
            result = model(batch.to(device))
            for key in outputs:
                outputs[key].append(result[key].detach().cpu().numpy())
            targets.append(target.numpy()); masks.append(mask.numpy())
    return {key: np.concatenate(value) for key, value in outputs.items()}, np.concatenate(targets), np.concatenate(masks)


def metric_row(true, pred, mask):
    return {
        "rmse": masked_rmse(true, pred, mask), "mae": masked_mae(true, pred, mask),
        "global_r2": masked_global_r2(true, pred, mask),
        "sample_pcc_median": median_per_sample_pcc(true, pred, mask),
        "protein_pcc_median": median_per_protein_pcc(true, pred, mask),
        "n_valid_positions": int((mask & np.isfinite(true) & np.isfinite(pred)).sum()),
    }


def level_norm_rows(structure, scope, output, mask):
    rows = []
    valid = np.asarray(mask, bool)
    for level in ("source", "instrument", "plate", "batch"):
        values = output[f"delta_{level}"] if level != "batch" else output["delta_batch"]
        selected = values[valid]
        rows.append({
            "structure": structure, "scope": scope, "level": level,
            "rms": float(np.sqrt(np.mean(np.square(selected, dtype=np.float64)))) if len(selected) else 0.0,
            "max_abs": float(np.max(np.abs(selected))) if len(selected) else 0.0,
        })
    return rows


def metadata_overlap(root, output):
    train_val_path = root / "WAYB_WAYC/WAYB_WAYC_metadata_train_val(1).csv"
    test_path = root / "WAYB_WAYC/WAYB_WAYC_metadata_test(1).csv"
    train_val = pd.read_csv(train_val_path)
    test = pd.read_csv(test_path)
    train = train_val.loc[train_val["split_final"].eq("train")]
    validation = train_val.loc[train_val["split_final"].astype(str).str.startswith("val_")]
    tuple_key = lambda frame: frame[list(BATCH_COLUMNS)].astype(str).agg("|".join, axis=1)
    result = {
        "train_val_metadata_path": str(train_val_path.resolve()), "train_val_metadata_sha256": sha256_file(train_val_path),
        "test_metadata_path": str(test_path.resolve()), "test_metadata_sha256": sha256_file(test_path),
        "test_metadata_only": True, "test_proteome_opened": False,
        "fields": list(BATCH_COLUMNS), "tuple_definition": list(BATCH_COLUMNS), "splits": {},
    }
    train_values = {column: set(train[column].astype(str)) for column in BATCH_COLUMNS}
    train_tuples = set(tuple_key(train))
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        result["splits"][name] = {
            "n_samples": len(frame),
            "unique_counts": {column: int(frame[column].astype(str).nunique()) for column in BATCH_COLUMNS},
            "unique_tuple_count": int(tuple_key(frame).nunique()),
            "unseen_vs_train": {
                column: sorted(set(frame[column].astype(str)) - train_values[column]) for column in BATCH_COLUMNS
            },
            "unseen_tuple_count_vs_train": int(len(set(tuple_key(frame)) - train_tuples)),
        }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run(config_path: Path, output_dir: Path, smoke=False):
    started = time.perf_counter(); config = load_config(config_path); seed = int(config["seed"])
    set_seed(seed); device = _resolve_device(config["training"]["device"])
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    artifacts = load_artifact_bundle(ROOT); meta = load_train_val_metadata(ROOT); labels, masks = load_label_frames(meta, artifacts, ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlap = metadata_overlap(ROOT, output_dir / "batch_metadata_overlap.json")
    train_ids = meta.index[meta["split_final"].eq("train")]
    control_names = meta.loc[train_ids, "perturbation_no_concentration"].astype(str).str.lower()
    controls = train_ids[control_names.isin(CONTROL_NAMES)]
    validation_controls = meta.index[
        meta["split_final"].astype(str).str.startswith("val_")
        & meta["perturbation_no_concentration"].astype(str).str.lower().isin(CONTROL_NAMES)
    ]
    chemical_variant = make_chemical_feature_variant(artifacts, "correct", seed)
    epochs = int(config["training"]["smoke"]["epochs"] if smoke else config["training"]["cv_epochs"])
    n_folds_run = int(config["training"]["smoke"]["folds"] if smoke else config["training"]["cv_folds"])
    fold_tables, all_metrics, norm_rows = {}, [], []

    for cv_name, column in GROUP_CV.items():
        folds = deterministic_group_folds(meta, controls, column, int(config["training"]["cv_folds"]), seed)
        fold_rows, split_rows = [], []
        for fold_spec in folds[:n_folds_run]:
            fold = fold_spec["fold"]
            fit_ids, inner_ids = split_inner_train(
                fold_spec["train_ids"], config["training"]["inner_validation_fraction"], seed, fold, "shared",
            )
            vocab = fit_category_vocabulary(meta, fit_ids)
            anchor_values, counts = fit_control_protein_anchor(meta, labels, masks, fit_ids)
            if counts.min() <= 0: raise ValueError("fold anchor has an unobserved protein")
            loaders = {}
            for role, ids, shuffle in (
                ("train", fit_ids, True), ("inner", inner_ids, False), ("holdout", fold_spec["holdout_ids"], False),
            ):
                batch = _batch_for(meta, ids, artifacts, vocab, chemical_variant, seed)
                loaders[role] = make_loader(
                    batch, torch.from_numpy(labels.loc[ids].to_numpy(np.float32, copy=True)),
                    torch.from_numpy(masks.loc[ids].to_numpy(bool, copy=True)),
                    config["training"]["batch_size"], shuffle, seed, config["training"]["num_workers"],
                )
            shared_initial_hashes = {}
            for structure in STRUCTURES:
                set_seed(seed)
                cfg = _model_config(config, artifacts, vocab, structure)
                model = StageABatchCalibrationModel(cfg, structure, torch.from_numpy(anchor_values)).to(device)
                shared_initial_hashes[structure] = module_parameters_sha256(model, ("genome_encoder", "baseline_branch"))
                run_dir = output_dir / f"{cv_name}_cv/fold_{fold}/{structure}"
                checkpoint = run_dir / "stage_a_best.pt"
                metadata = {
                    "stage": "S2A_stage_a_only", "structure": structure, "seed": seed,
                    "config": config, "artifact_hashes": artifacts.hashes,
                    "vocabulary": {key: list(value) for key, value in vocab.values.items()},
                    "fit_ids_sha256": _ids_hash(fit_ids), "inner_ids_sha256": _ids_hash(inner_ids),
                    "outer_holdout_ids_sha256": _ids_hash(fold_spec["holdout_ids"]),
                    "anchor_source": "outer_fold_inner_fit_train_controls_only",
                    "min_anchor_observations_per_protein": int(counts.min()),
                    "test_proteome_opened": False,
                }
                history, payload, reason = train_with_inner_early_stopping(
                    model, loaders["train"], loaders["inner"], config, epochs,
                    config["training"]["cv_patience"], checkpoint, metadata, device,
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
                output, true, mask = predict(model, loaders["holdout"], device)
                metrics = metric_row(true, output["y_pred"], mask)
                row = {
                    "cv_type": cv_name, "fold": fold, "structure": structure,
                    "train_sample_count": len(fit_ids), "inner_sample_count": len(inner_ids),
                    "holdout_sample_count": len(fold_spec["holdout_ids"]),
                    "train_group_count": int(meta.loc[fit_ids, column].astype(str).nunique()),
                    "holdout_group_count": len(fold_spec["holdout_groups"]),
                    "holdout_groups": "|".join(fold_spec["holdout_groups"]),
                    "group_overlap_count": 0, "best_epoch": int(payload["epoch"]),
                    "best_inner_huber": float(payload["monitor"]), "stop_reason": reason,
                    "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
                    "fit_ids_sha256": _ids_hash(fit_ids), "inner_ids_sha256": _ids_hash(inner_ids),
                    "holdout_ids_sha256": _ids_hash(fold_spec["holdout_ids"]),
                    "vocab_fit_scope": "outer_fold_inner_fit_controls_only",
                    "anchor_fit_scope": "outer_fold_inner_fit_controls_only",
                    "holdout_labels_used_for_training_or_selection": False,
                    **metrics,
                }
                fold_rows.append(row); all_metrics.append(row)
                norm_rows.extend(level_norm_rows(structure, f"{cv_name}_fold_{fold}_holdout", output, mask))
            if len(set(shared_initial_hashes.values())) != 1:
                raise RuntimeError("baseline/genome initialization differs across structures")
            split_rows.append({
                "cv_type": cv_name, "fold": fold, "column": column,
                "outer_train_ids_sha256": _ids_hash(fold_spec["train_ids"]),
                "fit_ids_sha256": _ids_hash(fit_ids), "inner_ids_sha256": _ids_hash(inner_ids),
                "holdout_ids_sha256": _ids_hash(fold_spec["holdout_ids"]),
                "holdout_groups": list(fold_spec["holdout_groups"]),
                "baseline_genome_initial_sha256": next(iter(shared_initial_hashes.values())),
                "group_overlap_count": 0,
            })
        frame = pd.DataFrame(fold_rows)
        frame.to_csv(output_dir / f"{cv_name}_group_cv_metrics.csv", index=False)
        (output_dir / f"{cv_name}_group_cv_splits.json").write_text(json.dumps(split_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        fold_tables[cv_name] = frame

    # Final Stage-A models: all 751 train controls, fixed epochs, no validation label selection.
    final_vocab = fit_category_vocabulary(meta, controls)
    final_anchor, final_counts = fit_control_protein_anchor(meta, labels, masks, controls)
    final_loaders = {}
    for role, ids, shuffle in (("train", controls, True), ("validation", validation_controls, False)):
        final_loaders[role] = make_loader(
            _batch_for(meta, ids, artifacts, final_vocab, chemical_variant, seed),
            torch.from_numpy(labels.loc[ids].to_numpy(np.float32, copy=True)), torch.from_numpy(masks.loc[ids].to_numpy(bool, copy=True)),
            config["training"]["batch_size"], shuffle, seed, config["training"]["num_workers"],
        )
    final_rows, oov_rows = [], []
    final_epochs = int(config["training"]["smoke"]["epochs"] if smoke else config["training"]["final_epochs"])
    for structure in STRUCTURES:
        set_seed(seed); cfg = _model_config(config, artifacts, final_vocab, structure)
        model = StageABatchCalibrationModel(cfg, structure, torch.from_numpy(final_anchor)).to(device)
        run_dir = output_dir / f"final/{structure}"; checkpoint = run_dir / "stage_a_final.pt"
        metadata = {
            "stage": "S2A_stage_a_only_final", "structure": structure, "seed": seed,
            "config": config, "artifact_hashes": artifacts.hashes,
            "vocabulary": {key: list(value) for key, value in final_vocab.values.items()},
            "fit_ids_sha256": _ids_hash(controls), "anchor_source": "all_train_controls_only",
            "control_sample_count": len(controls), "min_anchor_observations_per_protein": int(final_counts.min()),
            "validation_labels_used_for_training_or_selection": False, "test_proteome_opened": False,
        }
        history, payload = train_fixed_epochs(model, final_loaders["train"], config, final_epochs, checkpoint, metadata, device)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        output, true, mask = predict(model, final_loaders["validation"], device)
        final_rows.append({
            "structure": structure, "scope": "validation_controls", "n_samples": len(validation_controls),
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
            "epochs": final_epochs, "checkpoint_selection": "fixed_epochs_without_validation_labels",
            **metric_row(true, output["y_pred"], mask),
        })
        norm_rows.extend(level_norm_rows(structure, "validation_controls", output, mask))
        if structure == "hierarchical_batch":
            encoded = _batch_for(meta, validation_controls, artifacts, final_vocab, chemical_variant, seed)
            variants = {
                "known_full": encoded,
                "oov_plate": encoded.replace(batch_categorical=encoded.batch_categorical.clone()),
                "oov_instrument": encoded.replace(batch_categorical=encoded.batch_categorical.clone()),
                "all_oov": encoded.replace(batch_categorical=torch.zeros_like(encoded.batch_categorical)),
            }
            variants["oov_plate"].batch_categorical[:, 2] = 0
            variants["oov_instrument"].batch_categorical[:, 1:] = 0
            variant_outputs = {}
            for name, batch in variants.items():
                loader = make_loader(
                    batch, torch.from_numpy(labels.loc[validation_controls].to_numpy(np.float32, copy=True)),
                    torch.from_numpy(masks.loc[validation_controls].to_numpy(bool, copy=True)),
                    config["training"]["batch_size"], False, seed,
                )
                value, variant_true, variant_mask = predict(model, loader, device)
                variant_outputs[name] = value
                oov_rows.append({"oov_mode": name, **metric_row(variant_true, value["y_pred"], variant_mask)})
            full = variant_outputs["known_full"]
            checks = {
                "oov_plate_source_unchanged_max_abs": float(np.max(np.abs(full["delta_source"] - variant_outputs["oov_plate"]["delta_source"]))),
                "oov_plate_instrument_unchanged_max_abs": float(np.max(np.abs(full["delta_instrument"] - variant_outputs["oov_plate"]["delta_instrument"]))),
                "oov_plate_delta_plate_max_abs": float(np.max(np.abs(variant_outputs["oov_plate"]["delta_plate"]))),
                "oov_instrument_source_unchanged_max_abs": float(np.max(np.abs(full["delta_source"] - variant_outputs["oov_instrument"]["delta_source"]))),
                "oov_instrument_delta_instrument_max_abs": float(np.max(np.abs(variant_outputs["oov_instrument"]["delta_instrument"]))),
                "oov_instrument_delta_plate_max_abs": float(np.max(np.abs(variant_outputs["oov_instrument"]["delta_plate"]))),
                "all_oov_delta_batch_max_abs": float(np.max(np.abs(variant_outputs["all_oov"]["delta_batch"]))),
            }
    pd.DataFrame(final_rows).to_csv(output_dir / "validation_control_metrics.csv", index=False)
    pd.DataFrame(norm_rows).to_csv(output_dir / "hierarchical_level_norms.csv", index=False)
    pd.DataFrame(oov_rows).to_csv(output_dir / "oov_fallback_metrics.csv", index=False)
    (output_dir / "oov_fallback_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - started
    summary = {
        "status": "PASS", "mode": "smoke" if smoke else "formal", "seed": seed,
        "stage_b_trained": False, "chemical_response_trained": False, "fc_loss_enabled": False,
        "planning_proxy": True, "official_score": False, "device": device,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if device == "cuda" else 0,
        "n_train_controls": len(controls), "n_validation_controls": len(validation_controls),
        "feature_contract": {
            "n_proteins": artifacts.feature_contract.n_proteins,
            "protein_order_sha256": artifacts.feature_contract.protein_order_sha256,
            "generation_sha256": artifacts.feature_contract.generation_sha256,
        }, "artifact_hashes": artifacts.hashes,
        "metadata_overlap": overlap, "oov_checks": checks,
        "test_proteome_opened": False, "test_prediction_generated": False,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config, args.output_dir, args.smoke), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
