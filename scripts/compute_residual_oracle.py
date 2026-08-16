"""Compute residual (drug/context) oracle upper bounds, alongside the FC oracle.

Residual = Δ − μ, where μ is the drug-mean (μ_drug) or context-mean (μ_ctx).
The residual oracle measures whether the *specific* (background-subtracted)
perturbation response is more reproducible than the raw fold change.

Cross-plate pairs only (true biological replicates, different batch).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from baseline.data import load_raw_data, preprocess
from baseline.evaluation import (
    MATCH_KEYS,
    build_matched_control_pairs,
    build_train_residual_means,
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
    stats = build_train_residual_means(meta, y_log2, mask_matrix, train_mask)
    ctx_mean = stats["ctx_mean"]
    drug_mean = stats["drug_mean"]

    sids = list(fc_true.index)
    vals = fc_true.to_numpy(dtype=np.float64)   # (n_treat, P)
    msk = fc_mask.to_numpy(dtype=bool)
    pos = {sid: i for i, sid in enumerate(sids)}

    ctx_key = {sid: tuple(meta.loc[sid, k] for k in MATCH_KEYS) for sid in sids}
    drug_key = {sid: str(meta.loc[sid, "perturbation_no_concentration"]) for sid in sids}
    group_of = meta[GROUP_COLS].apply(tuple, axis=1).reindex(sids)
    plate_of = meta[PLATE_COLS].apply(tuple, axis=1).reindex(sids)

    # drug-mean residual (μ_drug includes self; n_drug is large so bias is negligible)
    def drug_residual(sid):
        i = pos[sid]
        mu = drug_mean.get(drug_key[sid])
        if mu is None:
            return None
        mu = np.asarray(mu, dtype=np.float64)
        valid = msk[i] & np.isfinite(mu)
        return vals[i] - mu, valid

    groups: dict = {}
    for sid in sids:
        groups.setdefault(group_of[sid], []).append(sid)

    fc_cross, drug_cross, ctx_cross, ctx_incl_cross = [], [], [], []
    for g, gsids in groups.items():
        if len(gsids) < 2:
            continue
        idxs = [pos[s] for s in gsids]
        ctx_vals = vals[idxs]
        ctx_mask = msk[idxs]
        ctx_sum = np.where(ctx_mask, ctx_vals, 0.0).sum(axis=0)
        ctx_cnt = ctx_mask.astype(np.float64).sum(axis=0)

        for a, b in combinations(gsids, 2):
            ia, ib = pos[a], pos[b]
            if plate_of[a] == plate_of[b]:
                continue  # cross-plate only

            # raw FC oracle
            v = msk[ia] & msk[ib]
            if int(v.sum()) >= 2:
                r = pcc(vals[ia][v], vals[ib][v])
                if np.isfinite(r):
                    fc_cross.append(r)

            # drug residual oracle
            ra = drug_residual(a)
            rb = drug_residual(b)
            if ra is not None and rb is not None:
                vv = ra[1] & rb[1]
                if int(vv.sum()) >= 2:
                    r = pcc(ra[0][vv], rb[0][vv])
                    if np.isfinite(r):
                        drug_cross.append(r)

            # context residual oracle (leave-one-out μ_ctx, exclude a and b)
            excl_sum = ctx_sum - np.where(msk[ia], vals[ia], 0.0) - np.where(msk[ib], vals[ib], 0.0)
            excl_cnt = ctx_cnt - msk[ia].astype(np.float64) - msk[ib].astype(np.float64)
            mu_loo = np.divide(excl_sum, excl_cnt, out=np.full_like(excl_sum, np.nan), where=excl_cnt > 0)
            va = msk[ia] & np.isfinite(mu_loo)
            vb = msk[ib] & np.isfinite(mu_loo)
            vv = va & vb
            if int(vv.sum()) >= 2:
                r = pcc((vals[ia] - mu_loo)[vv], (vals[ib] - mu_loo)[vv])
                if np.isfinite(r):
                    ctx_cross.append(r)

            # context residual oracle (含自身 μ_ctx, 官方口径, 虚假相关方向为 -1/(n-1))
            mu_incl = np.divide(ctx_sum, ctx_cnt, out=np.full_like(ctx_sum, np.nan), where=ctx_cnt > 0)
            vai = msk[ia] & np.isfinite(mu_incl)
            vbi = msk[ib] & np.isfinite(mu_incl)
            vvi = vai & vbi
            if int(vvi.sum()) >= 2:
                r = pcc((vals[ia] - mu_incl)[vvi], (vals[ib] - mu_incl)[vvi])
                if np.isfinite(r):
                    ctx_incl_cross.append(r)

    def _stat(name, arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            print(f"{name:32s} n=0")
            return
        print(
            f"{name:32s} n={arr.size:5d}  mean={arr.mean():.4f}  "
            f"median={np.median(arr):.4f}  p5={np.percentile(arr, 5):.4f}  "
            f"p95={np.percentile(arr, 95):.4f}"
        )

    print("=== 跨板号生物学重复 oracle 对比 ===")
    _stat("原始 FC (Δ)", fc_cross)
    _stat("drug 残差 (Δ - μ_drug)", drug_cross)
    _stat("ctx 残差 (Δ - μ_ctx, LOO)", ctx_cross)
    _stat("ctx 残差 (Δ - μ_ctx, 含自身)", ctx_incl_cross)


if __name__ == "__main__":
    main()
