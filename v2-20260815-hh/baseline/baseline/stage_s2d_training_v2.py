"""Stage S2D Stage-A-only retraining entry points for seed 20260815."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from baseline.baseline import hierarchical_batch_v2 as hb
from baseline.baseline.model_v2 import AnchoredVirtualCellV2, V2Config
from baseline.baseline.training_v2 import (
    BATCH_COLUMNS, CONTROL_NAMES, _loss_weights, _resolve_device, build_batch,
    configure_stage_optimizer, fit_category_vocabulary, fit_control_protein_anchor,
    load_artifact_bundle, load_config, load_label_frames, load_train_val_metadata,
    make_chemical_feature_variant, make_loader, run_training_stage, set_seed,
    sha256_file, stage_a_validation_control_ids,
)


ROOT = Path(__file__).resolve().parents[2]


def _ids_hash(ids):
    return hb._ids_hash(ids)


def train_hierarchical_stage_a(config_path: Path, output_dir: Path):
    started = time.perf_counter()
    config = load_config(config_path)
    seed = int(config["seed"])
    if seed != 20260815 or int(config["training"]["final_epochs"]) != 30:
        raise ValueError("S2D hierarchical Stage A requires seed 20260815 and 30 fixed epochs")
    set_seed(seed)
    device = _resolve_device(config["training"]["device"])
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    artifacts = load_artifact_bundle(ROOT)
    meta = load_train_val_metadata(ROOT)
    labels, masks = load_label_frames(meta, artifacts, ROOT)
    train_ids = meta.index[meta.split_final.eq("train")]
    names = meta.loc[train_ids, "perturbation_no_concentration"].astype(str).str.lower()
    controls = train_ids[names.isin(CONTROL_NAMES)]
    validation_controls = stage_a_validation_control_ids(meta)
    if len(controls) != 751 or len(validation_controls) != 205:
        raise ValueError("unexpected control counts")
    vocab = fit_category_vocabulary(meta, controls)
    anchor, counts = fit_control_protein_anchor(meta, labels, masks, controls)
    if int(counts.min()) <= 0:
        raise ValueError("control anchor has an unobserved protein")
    variant = make_chemical_feature_variant(artifacts, "correct", seed)
    cfg = hb._model_config(config, artifacts, vocab, "hierarchical_batch")
    model = hb.StageABatchCalibrationModel(cfg, "hierarchical_batch", torch.from_numpy(anchor)).to(device)
    batch = hb._batch_for(meta, controls, artifacts, vocab, variant, seed)
    loader = make_loader(
        batch, torch.from_numpy(labels.loc[controls].to_numpy(np.float32, copy=True)),
        torch.from_numpy(masks.loc[controls].to_numpy(bool, copy=True)),
        int(config["training"]["batch_size"]), True, seed, int(config["training"]["num_workers"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "stage_a_final.pt"
    metadata = {
        "stage": "S2A_stage_a_only_final", "structure": "hierarchical_batch", "seed": seed,
        "config": config, "artifact_hashes": artifacts.hashes,
        "vocabulary": {key: list(value) for key, value in vocab.values.items()},
        "fit_ids_sha256": _ids_hash(controls), "anchor_source": "all_train_controls_only",
        "control_sample_count": int(len(controls)),
        "min_anchor_observations_per_protein": int(counts.min()),
        "validation_labels_used_for_training_or_selection": False,
        "test_proteome_opened": False,
    }
    history, _ = hb.train_fixed_epochs(
        model, loader, config, 30, checkpoint, metadata, device,
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = {
        "status": "PASS", "kind": "hierarchical", "seed": seed, "device": device,
        "elapsed_seconds": time.perf_counter() - started, "fixed_epochs": 30,
        "checkpoint_selection": "fixed_epochs_without_validation_labels",
        "train_control_count": int(len(controls)), "validation_control_count_for_selection": 0,
        "fit_ids_sha256": _ids_hash(controls), "anchor_source": "all_train_controls_only",
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
        "artifact_hashes": artifacts.hashes, "test_proteome_opened": False,
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if device == "cuda" else 0,
    }
    (output_dir / "stage_a_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_flat_stage_a(config_path: Path, output_dir: Path):
    started = time.perf_counter()
    config = load_config(config_path)
    seed = int(config["training"]["seed"])
    stage_config = config["training"]["stage_a"]
    if seed != 20260815 or int(stage_config["epochs"]) != 30 or int(stage_config["patience"]) != 6:
        raise ValueError("S2D flat Stage A must reproduce the 30-epoch/patience-6 seed2 flow")
    set_seed(seed)
    device = _resolve_device(config["training"]["device"])
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    artifacts = load_artifact_bundle(ROOT)
    meta = load_train_val_metadata(ROOT)
    labels, masks = load_label_frames(meta, artifacts, ROOT)
    train_ids = meta.index[meta.split_final.eq("train")]
    vocab = fit_category_vocabulary(meta, train_ids)
    names = meta.loc[train_ids, "perturbation_no_concentration"].astype(str).str.lower()
    controls = train_ids[names.isin(CONTROL_NAMES)]
    validation_controls = stage_a_validation_control_ids(meta)
    anchor, counts = fit_control_protein_anchor(meta, labels, masks, controls)
    model_config = config["model"]
    cfg = V2Config(
        n_proteins=artifacts.feature_contract.n_proteins,
        latent_dim=int(model_config["latent_dim"]), protein_rank=int(model_config["protein_rank"]),
        dropout=float(model_config["dropout"]), batch_enabled=True, batch_structure="flat_batch",
        medium_vocab_size=vocab.size("Medium"),
        batch_vocab_sizes=tuple(vocab.size(column) for column in BATCH_COLUMNS),
    )
    model = AnchoredVirtualCellV2(cfg, torch.from_numpy(anchor)).to(device)
    variant = make_chemical_feature_variant(artifacts, model_config["chemical_mode"], seed)

    def loader(ids, shuffle):
        encoded = build_batch(
            meta, ids, artifacts, vocab, chemical_mode=model_config["chemical_mode"],
            genome_mode=model_config["genome_mode"], seed=seed, chemical_variant=variant,
        )
        return make_loader(
            encoded, torch.from_numpy(labels.loc[ids].to_numpy(np.float32, copy=True)),
            torch.from_numpy(masks.loc[ids].to_numpy(bool, copy=True)),
            int(config["training"]["batch_size"]), shuffle, seed,
            int(config["training"]["num_workers"]),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "stage_a_best.pt"
    optimizer = configure_stage_optimizer(model, stage_config, config["training"]["weight_decay"])
    metadata = {
        "anchor_source": "train_controls_only", "control_sample_count": int(len(controls)),
        "min_observations_per_protein": int(counts.min()),
        "fit_ids_sha256": _ids_hash(controls),
        "validation_control_ids_sha256": _ids_hash(validation_controls),
        "validation_labels_used_for_gradient": False, "test_proteome_opened": False,
    }
    result = run_training_stage(
        model, optimizer, loader(controls, True), loader(validation_controls, False),
        _loss_weights(config), int(stage_config["epochs"]), int(stage_config["patience"]),
        checkpoint, config, artifacts.hashes, seed, "control_first_A", device=device,
        monitor_kind="stage_a_control_only", run_metadata=metadata,
    )
    (output_dir / "history.json").write_text(json.dumps(list(result.history), indent=2), encoding="utf-8")
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = {
        "status": "PASS", "kind": "flat", "seed": seed, "device": device,
        "elapsed_seconds": time.perf_counter() - started,
        "best_epoch": result.best_epoch, "best_monitor": result.best_monitor,
        "actual_epochs": len(result.history), "stop_reason": result.stop_reason,
        "checkpoint_selection": "control_only_validation_huber",
        "train_control_count": int(len(controls)), "validation_control_count": int(len(validation_controls)),
        "fit_ids_sha256": _ids_hash(controls),
        "validation_control_ids_sha256": _ids_hash(validation_controls),
        "anchor_source": "train_controls_only", "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint), "artifact_hashes": artifacts.hashes,
        "test_proteome_opened": False,
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if device == "cuda" else 0,
    }
    (output_dir / "stage_a_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("hierarchical", "flat"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (
        train_hierarchical_stage_a(args.config, args.output_dir)
        if args.kind == "hierarchical"
        else train_flat_stage_a(args.config, args.output_dir)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
