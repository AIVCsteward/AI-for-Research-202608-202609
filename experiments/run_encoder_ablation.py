"""Train/evaluate the Person A encoder ablations with the Phase 1 MLP.

This is a reference runner for A6. It deliberately keeps the model side small and
stable so that the effect of each input feature group can be compared before
Person B's residual decoder is merged. Every encoder is fit on train rows only.

Example:
    python -m experiments.run_encoder_ablation --epochs 40 --output-dir experiments/outputs
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.data import get_split_masks, load_raw_data, preprocess
from baseline.evaluation import (
    build_control_lookup,
    build_matched_control_pairs,
    build_train_residual_means,
    evaluate_official_metrics,
    per_sample_corr,
)
from experiments.ablation_encoder import prepare_ablation_features

DEFAULT_EVAL_SPLITS = [
    "val_strain_only",
    "val_chem_only",
    "val_both",
    "val_time",
]

def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict_all(model, X_all, device, batch_size=2048):
    """Full-sample inference (needed so matched-control predictions are available)."""
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, X_all.shape[0], batch_size):
            x = torch.as_tensor(X_all[start:start + batch_size], dtype=torch.float32, device=device)
            out = model(x)
            if isinstance(out, dict):
                out = out["y_pred"]
            preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def _evaluate_model(
    model, X_all, meta, y_log2, mask_matrix, split_masks, splits, device,
    control_lookup, train_stats,
):
    pred_all = _predict_all(model, X_all, device)
    pred_df = pd.DataFrame(pred_all, index=meta.index, columns=y_log2.columns)
    metrics = {}
    for split_name in splits:
        split_mask = split_masks.get(split_name)
        if split_mask is None or int(split_mask.sum()) == 0:
            continue
        metrics[split_name] = evaluate_official_metrics(
            meta, y_log2, mask_matrix, pred_df, split_mask, control_lookup, train_stats
        )
    return metrics


def _make_sample_corr_monitor(
    X_tv, meta_tv, y_log2_tv, mask_tv, split_masks, split_name, device,
):
    """Return a ``monitor_fn(model)`` returning per-sample corr on a val split.

    Early stopping monitors the official "absolute fidelity" metric (per-sample
    Pearson correlation), which is stable and monotonically improvable — unlike
    FC PCC, which is noisy near zero and previously caused spurious early-stopping
    checkpoints (epoch-1 random weights) when used as the monitor.

    ``X_tv``/``meta_tv``/``y_log2_tv``/``mask_tv`` are the train+val-only subset
    (test is excluded so early stopping does not leak test), aligned to one
    another.  ``split_masks`` is indexed by the full metadata.
    """
    split_mask = split_masks[split_name].loc[meta_tv.index].astype(bool)
    y_true_sub = y_log2_tv.loc[split_mask]
    mask_sub = mask_tv.loc[split_mask]

    def monitor_fn(model):
        pred_all = _predict_all(model, X_tv, device)
        pred_df = pd.DataFrame(pred_all, index=meta_tv.index, columns=y_log2_tv.columns)
        pred_sub = pred_df.loc[split_mask].to_numpy(dtype=np.float64)
        return per_sample_corr(y_true_sub, pred_sub, mask_sub)

    return monitor_fn


def run_ablation(
    meta,
    y_log2,
    mask_matrix,
    protein_count: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device,
    eval_splits,
    use_residual_loss: bool = False,
    model_type: str = "mlp",
    groups=None,
):
    import torch

    from baseline.model import ConditionMLP
    from aivc.training import (
        build_residual_mean_tensors,
        prepare_fold_change_index,
        prepare_training_data,
        train,
    )

    split_masks = get_split_masks(meta)
    features = prepare_ablation_features(meta, y_log2, mask_matrix)
    if groups:
        features = {k: v for k, v in features.items() if k in groups}
    train_mask = meta["split_final"].astype(str).eq("train")
    control_lookup, _, _ = build_control_lookup(meta, y_log2, train_mask=None)
    train_stats = build_train_residual_means(meta, y_log2, mask_matrix, train_mask)

    # train+val-only subset for the sample-corr early-stopping monitor (no test leakage).
    tv_mask = ~meta["split_final"].astype(str).str.startswith("test")
    meta_tv = meta.loc[tv_mask]
    y_log2_tv = y_log2.loc[meta_tv.index]
    mask_tv = mask_matrix.loc[meta_tv.index]

    fc_control_index = None
    ctx_t = None
    drug_t = None
    loss_weights = None
    if use_residual_loss:
        pairs = build_matched_control_pairs(meta, train_mask)
        fc_control_index = prepare_fold_change_index(
            meta.index[train_mask].tolist(), pairs, device=device
        )
        ctx_t, drug_t = build_residual_mean_tensors(
            meta, y_log2, mask_matrix, train_mask, device=device
        )
        loss_weights = {"mse": 1.0, "fc": 1.0, "ctx": 0.5, "drug": 0.5, "l2": 0.01, "corr": 0.0}

    results = {}

    for name, bundle in features.items():
        print(f"\n[A6] {name}: X={bundle['X'].shape}, raw={bundle['encoders']['raw_dim']}")
        _set_seed(seed)
        if model_type == "aivc":
            from aivc.model import AIVCModel

            model = AIVCModel(bundle["X"].shape[1], protein_count, dim_emb=256, use_gnn=False)
        else:
            model = ConditionMLP(bundle["X"].shape[1], protein_count, hidden=256, dropout=0.1)
        model = model.to(device)
        x_train, y_train, mask_train, val_data = prepare_training_data(
            bundle["X"],
            y_log2,
            mask_matrix,
            train_mask,
            split_masks,
            [s for s in eval_splits if s.startswith("val_")],
            device,
        )
        monitor_fn = None
        if use_residual_loss:
            monitor_fn = _make_sample_corr_monitor(
                bundle["X"][tv_mask.values],
                meta_tv,
                y_log2_tv,
                mask_tv,
                split_masks,
                "val_both",
                device,
            )
        model, history = train(
            model,
            x_train,
            y_train,
            mask_train,
            val_data,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            verbose=True,
            fc_control_index=fc_control_index,
            ctx_mean_t=ctx_t,
            drug_mean_t=drug_t,
            loss_weights=loss_weights,
            monitor_fn=monitor_fn,
        )
        val_losses = [value for values in history["val_loss"].values() for value in values]
        results[name] = {
            "feature_shape": list(bundle["X"].shape),
            "raw_dim": int(bundle["encoders"]["raw_dim"]),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "best_val_loss": float(min(val_losses)) if val_losses else None,
            "metrics": _evaluate_model(
                model,
                bundle["X"],
                meta,
                y_log2,
                mask_matrix,
                split_masks,
                eval_splits,
                device,
                control_lookup,
                train_stats,
            ),
        }
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Person A encoder ablations")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu, cuda, or omit for auto")
    parser.add_argument(
        "--model", default="mlp", choices=["mlp", "aivc"],
        help="Model: mlp (ConditionMLP) or aivc (AIVCModel residual decoder)",
    )
    parser.add_argument(
        "--groups", nargs="*", default=None,
        help="Which ablation groups to run (default: all). e.g. full no_chemical_structure",
    )
    parser.add_argument(
        "--use-residual-loss", action="store_true",
        help="Add FC + context/drug residual Pearson losses on top of MSE",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/outputs")
    )
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=DEFAULT_EVAL_SPLITS,
        help="Splits to evaluate; default is the four validation buckets",
    )
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")

    import torch

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"加载数据，设备={device}")
    meta, prot = load_raw_data()
    y_log2, mask_matrix, meta, protein_names, _ = preprocess(meta, prot)
    results = run_ablation(
        meta,
        y_log2,
        mask_matrix,
        protein_count=len(protein_names),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
        eval_splits=args.eval_splits,
        use_residual_loss=args.use_residual_loss,
        model_type=args.model,
        groups=args.groups,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "encoder_ablation_results.json"

    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
            return None
        return obj

    output_path.write_text(
        json.dumps(_sanitize(results), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nA6 完成，结果已写入: {output_path}")

    # Human-readable summary of the official metrics per group per split.
    def _f(x):
        return "nan" if x is None or (isinstance(x, float) and x != x) else f"{x:.4f}"

    print("\n=== 官方指标摘要 ===")
    for name, res in results.items():
        for split_name in ("val_chem_only", "val_strain_only", "val_both", "val_time"):
            m = res["metrics"].get(split_name)
            if not m:
                continue
            print(
                f"{name:22s} {split_name:16s} "
                f"sample_corr={m['per_sample_corr']:.4f} fc_pcc={_f(m['fc_pcc'])} "
                f"ctx_res={_f(m['context_residual_pcc'])} drug_res={_f(m['drug_residual_pcc'])} "
                f"dir_acc={_f(m['high_effect_dir_acc'])} ppr2={m['per_protein_r2_median']:.4f}"
            )


if __name__ == "__main__":
    main()


