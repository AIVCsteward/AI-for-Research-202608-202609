"""Evaluate the two no-model baselines under the official six-module metrics.

Baselines:
  - Protein Mean: predict every sample with the train-set per-protein mean.
  - Matched Control: predict every sample with its matched Water/DMSO control.

Outputs the full official metric dict per validation split for each baseline,
so they can sit alongside the model ablation tables in the experiment log.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from baseline.data import get_split_masks, load_raw_data, preprocess
from baseline.evaluation import (
    VAL_SPLITS,
    build_control_lookup,
    build_train_residual_means,
    compute_protein_mean,
    evaluate_official_metrics,
    matched_control_predict,
)


def _to_df(pred_array, sample_index, protein_names):
    return pd.DataFrame(pred_array, index=sample_index, columns=protein_names)


def main():
    meta, prot = load_raw_data()
    y_log2, mask_matrix, meta, protein_names, _ = preprocess(meta, prot)
    split_masks = get_split_masks(meta)

    train_mask = meta["split_final"].astype(str).eq("train")
    protein_mean = compute_protein_mean(y_log2, train_mask)
    # Full-data control lookup (matched controls are real measurements, not train-only).
    control_lookup, control_mean, _ = build_control_lookup(meta, y_log2, train_mask=None)
    train_stats = build_train_residual_means(meta, y_log2, mask_matrix, train_mask)

    results = {}
    for split_name in VAL_SPLITS:
        m = split_masks.get(split_name)
        if m is None or int(m.sum()) == 0:
            continue
        idx = meta.loc[m].index
        n = int(m.sum())

        # Protein Mean baseline
        pm_pred = np.tile(protein_mean.values, (n, 1))
        pm_metrics = evaluate_official_metrics(
            meta, y_log2, mask_matrix, _to_df(pm_pred, idx, protein_names),
            m, control_lookup, train_stats,
        )

        # Matched Control baseline
        mc_pred = matched_control_predict(
            meta.loc[m], y_log2, control_lookup, protein_mean, control_mean
        )
        mc_metrics = evaluate_official_metrics(
            meta, y_log2, mask_matrix, _to_df(mc_pred, idx, protein_names),
            m, control_lookup, train_stats,
        )

        results[split_name] = {
            "protein_mean": pm_metrics,
            "matched_control": mc_metrics,
        }

    # Human-readable table
    cols = ["sample_corr", "fc_pcc", "ctx_res", "drug_res", "dir_acc", "ppr2"]

    def _f(x):
        return "nan" if x is None or (isinstance(x, float) and x != x) else f"{x:.4f}"

    print(f"{'split':16s} {'baseline':16s} " + " ".join(f"{c:>10s}" for c in cols))
    for split_name, bl in results.items():
        for base_name in ("protein_mean", "matched_control"):
            mm = bl[base_name]
            row = " ".join(
                _f(mm.get(
                    "per_sample_corr" if c == "sample_corr" else
                    "fc_pcc" if c == "fc_pcc" else
                    "context_residual_pcc" if c == "ctx_res" else
                    "drug_residual_pcc" if c == "drug_res" else
                    "high_effect_dir_acc" if c == "dir_acc" else
                    "per_protein_r2_median"
                ))
                for c in cols
            )
            print(f"{split_name:16s} {base_name:16s} {row}")

    # JSON for the log
    print("\n=== JSON ===")
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
