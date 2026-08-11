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

from baseline.data import get_split_masks, load_raw_data, preprocess
from experiments.ablation_encoder import prepare_ablation_features

DEFAULT_EVAL_SPLITS = [
    "val_strain_only",
    "val_chem_only",
    "val_both",
    "val_time",
]

def evaluate_global_r2(y_true, y_pred, mask):
    y_t = y_true.fillna(0).to_numpy()
    m = mask.to_numpy(dtype=float)
    ss_res = ((y_t - y_pred) ** 2 * m).sum()
    grand_mean = (y_t * m).sum() / max(m.sum(), 1.0)
    ss_tot = ((y_t - grand_mean) ** 2 * m).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def evaluate_per_protein_r2(y_true, y_pred, mask):
    y_t = y_true.to_numpy()
    m = mask.to_numpy(dtype=bool)
    scores = []
    for column in range(y_t.shape[1]):
        valid = m[:, column]
        if valid.sum() < 3:
            continue
        observed = y_t[valid, column]
        ss_tot = ((observed - observed.mean()) ** 2).sum()
        if ss_tot > 0:
            ss_res = ((observed - y_pred[valid, column]) ** 2).sum()
            scores.append(1.0 - ss_res / ss_tot)
    return float(np.median(scores)) if scores else float("nan")

def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate_model(model, X_all, meta, y_log2, mask_matrix, split_masks, splits, device):
    import torch

    model.eval()
    metrics = {}
    with torch.no_grad():
        for split_name in splits:
            split_mask = split_masks.get(split_name)
            if split_mask is None or int(split_mask.sum()) == 0:
                continue
            row_mask = split_mask.to_numpy(dtype=bool)
            x = torch.as_tensor(X_all[row_mask], dtype=torch.float32, device=device)
            pred = model(x).detach().cpu().numpy()
            y_true = y_log2.loc[split_mask]
            observed = mask_matrix.loc[split_mask]
            metrics[split_name] = {
                "n_samples": int(row_mask.sum()),
                "global_r2": float(evaluate_global_r2(y_true, pred, observed)),
                "per_protein_r2_median": float(
                    evaluate_per_protein_r2(y_true, pred, observed)
                ),
            }
    return metrics


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
):
    import torch

    from baseline.model import ConditionMLP
    from aivc.training import prepare_training_data, train

    split_masks = get_split_masks(meta)
    features = prepare_ablation_features(meta, y_log2, mask_matrix)
    train_mask = meta["split_final"].astype(str).eq("train")
    results = {}

    for name, bundle in features.items():
        print(f"\n[A6] {name}: X={bundle['X'].shape}, raw={bundle['encoders']['raw_dim']}")
        _set_seed(seed)
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
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "encoder_ablation_results.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nA6 完成，结果已写入: {output_path}")


if __name__ == "__main__":
    main()


