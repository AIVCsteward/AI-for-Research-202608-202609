"""验证去噪（低秩 + 高SNR）能否抬高 fc_pcc。

对比 4 个配置（ConditionMLP + residual loss，只动 FC 目标）：
  baseline / lowrank / snr / lowrank+snr
训练用去噪目标，评估仍用原始 FC（官方口径）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.data import get_split_masks, load_raw_data, preprocess
from baseline.evaluation import (
    build_control_lookup,
    build_matched_control_pairs,
    build_train_residual_means,
    evaluate_official_metrics,
)
from baseline.model import ConditionMLP
from experiments.ablation_encoder import prepare_ablation_features
from aivc.training import (
    build_residual_mean_tensors,
    prepare_fold_change_index,
    prepare_training_data,
    train,
)

EVAL_SPLITS = ["val_strain_only", "val_chem_only", "val_both", "val_time"]


def _predict_all(model, X_all, device, batch_size=2048):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="mlp", choices=["mlp", "aivc"])
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"设备={device}")

    meta, prot = load_raw_data()
    y_log2, mask_matrix, meta, protein_names, _ = preprocess(meta, prot)
    split_masks = get_split_masks(meta)
    features = prepare_ablation_features(meta, y_log2, mask_matrix)
    X = features["full"]["X"]

    train_mask = meta["split_final"].astype(str).eq("train")
    control_lookup, _, _ = build_control_lookup(meta, y_log2, train_mask=None)
    train_stats = build_train_residual_means(meta, y_log2, mask_matrix, train_mask)

    pairs = build_matched_control_pairs(meta, train_mask)
    fc_control_index = prepare_fold_change_index(
        meta.index[train_mask].tolist(), pairs, device=device
    )
    ctx_t, drug_t = build_residual_mean_tensors(
        meta, y_log2, mask_matrix, train_mask, device=device
    )
    loss_weights = {"mse": 1.0, "fc": 1.0, "ctx": 0.5, "drug": 0.5, "l2": 0.01, "corr": 0.0}

    # 去噪目标（多 k）
    K_LIST = [10, 20, 25, 30]
    lowrank_targets = {
        k: torch.as_tensor(
            np.load(f"experiments/outputs/oracle/denoised_fc_target_k{k}.npy"),
            dtype=torch.float32, device=device,
        )
        for k in K_LIST
    }

    x_train, y_train, mask_train, val_data = prepare_training_data(
        X, y_log2, mask_matrix, train_mask, split_masks, EVAL_SPLITS, device
    )

    configs = {"baseline": (None, None)}
    for k in K_LIST:
        configs[f"lowrank_k{k}"] = (lowrank_targets[k], None)

    results = {}
    for name, (d_t, s_t) in configs.items():
        print(f"\n===== {name} =====")
        torch.manual_seed(42)
        if args.model == "aivc":
            from aivc.model import AIVCModel
            model = AIVCModel(X.shape[1], len(protein_names), dim_emb=256, use_gnn=False).to(device)
        else:
            model = ConditionMLP(X.shape[1], len(protein_names), hidden=256, dropout=0.1).to(device)
        model, history = train(
            model, x_train, y_train, mask_train, val_data,
            epochs=args.epochs, batch_size=256, lr=1e-3, weight_decay=1e-5,
            device=device, verbose=False,
            fc_control_index=fc_control_index, ctx_mean_t=ctx_t, drug_mean_t=drug_t,
            loss_weights=loss_weights, denoised_fc_target=d_t, snr_mask=s_t,
        )
        pred_all = _predict_all(model, X, device)
        pred_df = pd.DataFrame(pred_all, index=meta.index, columns=y_log2.columns)
        metrics = {}
        for split in EVAL_SPLITS:
            m = split_masks[split]
            if int(m.sum()) == 0:
                continue
            metrics[split] = evaluate_official_metrics(
                meta, y_log2, mask_matrix, pred_df, m, control_lookup, train_stats
            )
        results[name] = metrics

    # 汇总 fc_pcc
    print("\n===== fc_pcc 对比（官方口径，原始 FC）=====")
    summary = {}
    for name, metrics in results.items():
        row = {s: metrics[s]["fc_pcc"] for s in EVAL_SPLITS if s in metrics}
        summary[name] = row
        print(f"{name:14s} " + "  ".join(f"{s}={row.get(s)}" for s in EVAL_SPLITS))

    out = Path("experiments/outputs/oracle/denoise_fc_pcc.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
