"""Stage 2 multi-objective runner for Person C integration and smoke tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aivc.config import get_experiment_config
from baseline.data import get_split_masks, preprocess
from baseline.evaluation import VAL_SPLITS, build_matched_control_pairs
from baseline.features import build_condition_features, fit_feature_encoders
from aivc.model import AIVCModel
from aivc.training import prepare_fold_change_index, train


def load_train_val(data_dir):
    metadata_path = data_dir / "WAYB_WAYC_metadata_train_val(1).csv"
    proteome_path = data_dir / "WAYB_WAYC_proteome_raw_train_val.csv"
    meta = pd.read_csv(metadata_path)
    protein_columns = pd.read_csv(proteome_path, nrows=0).columns.tolist()
    dtypes = {column: np.float32 for column in protein_columns if column != "sample_ID"}
    proteome = pd.read_csv(proteome_path, dtype=dtypes)
    return meta, proteome


def select_smoke_training_ids(meta, pairs, treatment_count):
    smoke_pairs = pairs.iloc[:treatment_count].copy()
    selected = set(smoke_pairs["treatment_sample_ID"])
    for control_ids in smoke_pairs["control_sample_ids"]:
        selected.update(control_ids)
    ordered_ids = [sample_id for sample_id in meta.index if sample_id in selected]
    return ordered_ids, smoke_pairs


def tensors_for_ids(X, y_log2, mask_matrix, sample_ids, device):
    if X.shape[0] != len(sample_ids):
        raise ValueError("Feature rows and sample_ID count do not match")
    return (
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(y_log2.loc[sample_ids].fillna(0).values, dtype=torch.float32, device=device),
        torch.tensor(mask_matrix.loc[sample_ids].values, dtype=torch.float32, device=device),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-treatment-count", type=int, default=64)
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device)
    config = get_experiment_config({"gnn": {"enabled": False}})

    raw_meta, raw_proteome = load_train_val(args.data_dir)
    y_log2, mask_matrix, meta, protein_names, train_mask = preprocess(
        raw_meta, raw_proteome
    )
    split_masks = get_split_masks(meta)

    pairs = build_matched_control_pairs(meta, train_mask)
    if args.smoke:
        train_sample_ids, pairs_for_training = select_smoke_training_ids(
            meta, pairs, args.smoke_treatment_count
        )
    else:
        train_sample_ids = meta.index[train_mask].tolist()
        pairs_for_training = pairs

    train_meta = meta.loc[train_sample_ids]
    encoders = fit_feature_encoders(
        train_meta,
        y_log2.loc[train_sample_ids],
        mask_matrix.loc[train_sample_ids],
        config=config,
    )
    X_train = build_condition_features(train_meta, encoders=encoders)
    X_train_t, y_train_t, mask_train_t = tensors_for_ids(
        X_train, y_log2, mask_matrix, train_sample_ids, device
    )
    fc_control_index = prepare_fold_change_index(
        train_sample_ids, pairs_for_training, device=device
    )

    val_data = {}
    for split_name in VAL_SPLITS:
        if split_name not in split_masks:
            continue
        val_ids = meta.index[split_masks[split_name]].tolist()
        if args.smoke:
            val_ids = val_ids[:32]
        if not val_ids:
            continue
        X_val_np = build_condition_features(meta.loc[val_ids], encoders=encoders)
        X_val, y_val, mask_val = tensors_for_ids(
            X_val_np, y_log2, mask_matrix, val_ids, device
        )
        val_data[split_name] = {"X": X_val, "y_filled": y_val, "mask_t": mask_val}

    model = AIVCModel(
        dim_in=X_train_t.shape[1],
        n_proteins=len(protein_names),
        dim_emb=config["encoder"]["d_emb"],
        use_gnn=False,
    ).to(device)
    loss_weights = {
        "mse": config["loss"]["mse_weight"],
        "fc": config["loss"]["fc_pearson_weight"],
        "l2": config["loss"]["residual_l2_weight"],
        "corr": 0.0,
    }

    print(f"device={device}")
    print(f"train rows={len(train_sample_ids)}, matched treatments={len(pairs_for_training)}")
    print(f"proteins={len(protein_names)}, feature dim={X_train_t.shape[1]}")
    model, history = train(
        model,
        X_train_t,
        y_train_t,
        mask_train_t,
        val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        verbose=True,
        fc_control_index=fc_control_index,
        loss_weights=loss_weights,
        early_stopping_patience=config["training"]["early_stopping_patience"],
        early_stopping_split="val_both",
    )
    del model

    print("final loss components:")
    for key in ("loss_total", "loss_mse", "loss_fc", "loss_l2", "loss_corr"):
        print(f"  {key}={history[key][-1]:.6f}")
    print(f"  val_both_per_protein_r2={history['val_per_protein_r2']['val_both'][-1]:.6f}")
    print("Stage 2 smoke test completed; no test truth was read.")


if __name__ == "__main__":
    main()
