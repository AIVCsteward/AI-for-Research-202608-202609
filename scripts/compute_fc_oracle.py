"""Compute the FC oracle upper bound: biological-replicate FC consistency.

For each biological condition (strain x medium x temp x time x drug) with
>= 2 treatment replicates in train, compute PCC between every pair of
replicate fold-changes.  "Copy another replicate of the same condition" is the
strongest possible FC predictor, so this bounds fc_pcc from above.

Also splits replicate pairs into:
  - same-plate (technical replicates, same batch) — tends to over-estimate signal
  - cross-plate (true biological replicates, different batch) — the honest ceiling
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from baseline.data import load_raw_data, preprocess
from baseline.evaluation import (
    build_matched_control_pairs,
    compute_fold_change,
    pcc,
)

GROUP_COLS = ["Strains", "Medium", "Temperature", "pert_time", "perturbation_no_concentration"]
PLATE_COLS = ["data_source", "instrument", "Yeast_cell_plate"]


def main():
    meta, prot = load_raw_data()
    y_log2, mask_matrix, meta, protein_names, _ = preprocess(meta, prot)
    train_mask = meta["split_final"].astype(str).eq("train")

    pairs = build_matched_control_pairs(meta, train_mask)
    fc = compute_fold_change(meta, y_log2, mask_matrix, train_mask, pairs)
    fc_true = fc["fc_true"]
    fc_mask = fc["fc_mask"]

    vals = fc_true.to_numpy(dtype=np.float64)
    msk = fc_mask.to_numpy(dtype=bool)
    pos = {sid: i for i, sid in enumerate(fc_true.index)}

    group_of = meta[GROUP_COLS].apply(tuple, axis=1).reindex(fc_true.index)
    plate_of = meta[PLATE_COLS].apply(tuple, axis=1).reindex(fc_true.index)

    groups: dict = {}
    for sid in fc_true.index:
        groups.setdefault(group_of[sid], []).append(sid)

    all_pccs = []
    same_plate_pccs = []
    cross_plate_pccs = []
    per_group_mean = []
    per_group_n = []

    for g, sids in groups.items():
        if len(sids) < 2:
            continue
        g_pccs = []
        for a, b in combinations(sids, 2):
            i, j = pos[a], pos[b]
            valid = msk[i] & msk[j]
            if int(valid.sum()) < 2:
                continue
            r = pcc(vals[i][valid], vals[j][valid])
            if not np.isfinite(r):
                continue
            g_pccs.append(r)
            all_pccs.append(r)
            if plate_of[a] == plate_of[b]:
                same_plate_pccs.append(r)
            else:
                cross_plate_pccs.append(r)
        if g_pccs:
            per_group_mean.append(float(np.mean(g_pccs)))
            per_group_n.append(len(sids))

    all_pccs = np.asarray(all_pccs)
    same_plate_pccs = np.asarray(same_plate_pccs)
    cross_plate_pccs = np.asarray(cross_plate_pccs)
    per_group_mean = np.asarray(per_group_mean)

    def _stat(name, arr):
        if arr.size == 0:
            print(f"{name:28s} (n=0)")
            return
        print(
            f"{name:28s} n={arr.size:5d}  "
            f"mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
            f"p5={np.percentile(arr, 5):.4f}  p95={np.percentile(arr, 95):.4f}"
        )

    print(f"train treatment 样本: {len(fc_true.index)}, 有重复的组合: {len(per_group_mean)}")
    print()
    print("=== FC oracle 上界 ===")
    _stat("所有重复对 (平权)", all_pccs)
    _stat("  同板号 (技术重复)", same_plate_pccs)
    _stat("  跨板号 (生物学重复)", cross_plate_pccs)
    _stat("每组合代表PCC (跨组合平均)", per_group_mean)
    print()

    # 按组合重复次数分层
    print("=== 按组合重复次数分层 (每组合代表PCC) ===")
    df = pd.DataFrame({"n": per_group_n, "pcc": per_group_mean})
    for n, sub in df.groupby("n"):
        print(
            f"  重复数={n}: 组合数={len(sub):4d}  mean_pcc={sub['pcc'].mean():.4f}  "
            f"median_pcc={sub['pcc'].median():.4f}"
        )


if __name__ == "__main__":
    main()
