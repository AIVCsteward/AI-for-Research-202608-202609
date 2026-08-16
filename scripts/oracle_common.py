"""Shared cache + oracle helpers for the four SNR experiments.

The precompute step writes a read-only cache (numpy arrays + metadata) so the
four plan scripts never reload the raw 290MB proteome CSV.  Every plan reads
this cache (read-only) and writes its own output JSON — so they can run in
parallel without conflicts.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("data/external/oracle_cache")
CACHE_NPZ = CACHE_DIR / "cache.npz"
CACHE_META = CACHE_DIR / "meta.pkl"
CACHE_GROUP = CACHE_DIR / "group_to_samples.json"

# 生物条件分组键（不含测量字段）；plate 键用于区分技术/生物学重复
GROUP_COLS = ["Strains", "Medium", "Temperature", "pert_time", "perturbation_no_concentration"]
PLATE_COLS = ["data_source", "instrument", "Yeast_cell_plate"]

DEFAULT_K = [500, 1000, 2000, 4422]


def cache_exists() -> bool:
    return CACHE_NPZ.exists() and CACHE_META.exists() and CACHE_GROUP.exists()


def load_cache() -> dict:
    if not cache_exists():
        raise RuntimeError(
            "缓存不存在，请先运行: python scripts/precompute_oracle_cache.py"
        )
    z = np.load(CACHE_NPZ, allow_pickle=True)
    meta = pd.read_pickle(CACHE_META)
    with open(CACHE_GROUP, "r", encoding="utf-8") as f:
        group_to_samples = json.load(f)
    # group_to_samples 的 key 是 str(tuple)，value 是 treat 行索引列表
    group_to_samples = {k: [int(i) for i in v] for k, v in group_to_samples.items()}
    return {
        "y_log2": z["y_log2"],           # (n_samples, n_protein) float32, NaN 无效
        "mask": z["mask"],               # (n_samples, n_protein) bool
        "protein_names": z["protein_names"],
        "fc_true": z["fc_true"],         # (n_treat, n_protein) float32, NaN 无效
        "fc_mask": z["fc_mask"],         # (n_treat, n_protein) bool
        "treat_index": z["treat_index"],  # (n_treat,) treatment sample_ID
        "replicate_pairs": z["replicate_pairs"],  # (n_pairs, 3): [idx_a, idx_b, is_cross_plate]
        "meta": meta,
        "group_to_samples": group_to_samples,
    }


def build_cache() -> None:
    """Generate the read-only cache.  Runs once; idempotent (re-writes atomically)."""
    from baseline.data import load_raw_data, preprocess
    from baseline.evaluation import build_matched_control_pairs, compute_fold_change

    meta, prot = load_raw_data()
    y_log2, mask_matrix, meta, protein_names, _ = preprocess(meta, prot)
    train_mask = meta["split_final"].astype(str).eq("train")

    pairs = build_matched_control_pairs(meta, train_mask)
    fc = compute_fold_change(meta, y_log2, mask_matrix, train_mask, pairs)
    fc_true = fc["fc_true"]
    fc_mask = fc["fc_mask"]

    y_np = y_log2.to_numpy(dtype=np.float32)
    mask_np = mask_matrix.to_numpy(dtype=bool)
    fc_np = fc_true.to_numpy(dtype=np.float32)
    fcm_np = fc_mask.to_numpy(dtype=bool)
    treat_index = fc_true.index.to_numpy()

    # treat 行索引与 sample_ID 的映射，以及分组 / plate 信息
    group_of = meta[GROUP_COLS].apply(tuple, axis=1).reindex(treat_index)
    plate_of = meta[PLATE_COLS].apply(tuple, axis=1).reindex(treat_index)
    pos = {sid: i for i, sid in enumerate(treat_index)}

    groups: dict = {}
    for sid in treat_index:
        groups.setdefault(str(group_of[sid]), []).append(int(pos[sid]))

    rep_pairs = []
    for idxs in groups.values():
        for a, b in combinations(idxs, 2):
            is_cross = int(plate_of.iloc[a] != plate_of.iloc[b])
            rep_pairs.append([a, b, is_cross])
    replicate_pairs = np.asarray(rep_pairs, dtype=np.int64) if rep_pairs else np.zeros((0, 3), dtype=np.int64)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # 原子写：先写临时文件再替换，避免多进程同时写时读到半截文件
    tmp = CACHE_NPZ.with_name("cache.tmp.npz")
    np.savez_compressed(
        tmp,
        y_log2=y_np, mask=mask_np, protein_names=np.asarray(protein_names),
        fc_true=fc_np, fc_mask=fcm_np, treat_index=treat_index,
        replicate_pairs=replicate_pairs,
    )
    tmp.replace(CACHE_NPZ)
    meta.to_pickle(CACHE_META)
    with open(CACHE_GROUP, "w", encoding="utf-8") as f:
        json.dump(groups, f)
    print(f"缓存已生成: {CACHE_DIR}")
    print(f"  y_log2={y_np.shape}, fc_true={fc_np.shape}, replicate_pairs={replicate_pairs.shape}")
    tech = int((replicate_pairs[:, 2] == 0).sum())
    bio = int((replicate_pairs[:, 2] == 1).sum())
    print(f"  技术重复对(同板)={tech}, 生物学重复对(跨板)={bio}")


def cross_plate_oracle(fc_matrix, fc_mask, replicate_pairs, only_cross_plate=True):
    """对 (去噪后的) FC 矩阵算跨板号生物学重复的 FC 一致性 oracle。

    fc_matrix / fc_mask: (n_treat, n_protein)，行序与 replicate_pairs 的 idx 对齐。
    返回 None 或 {"n","mean","median","p5","p95"}。
    """
    from baseline.evaluation import pcc

    pccs = []
    for i, j, is_cross in replicate_pairs:
        if only_cross_plate and not is_cross:
            continue
        valid = fc_mask[i] & fc_mask[j]
        if int(valid.sum()) < 2:
            continue
        r = pcc(fc_matrix[i][valid], fc_matrix[j][valid])
        if np.isfinite(r):
            pccs.append(float(r))
    pccs = np.asarray(pccs)
    if pccs.size == 0:
        return None
    return {
        "n": int(pccs.size),
        "mean": float(pccs.mean()),
        "median": float(np.median(pccs)),
        "p5": float(np.percentile(pccs, 5)),
        "p95": float(np.percentile(pccs, 95)),
    }


def masked_var_across_pairs(diff, valid):
    """对 (n_pairs, n_protein) 的 diff/valid，沿 pair 轴算每蛋白的 mask-aware 方差。"""
    cnt = valid.sum(axis=0).astype(np.float64)
    safe = np.where(cnt > 0, cnt, 1.0)
    mean = diff.sum(axis=0) / safe
    centered = np.where(valid, diff - mean[None, :], 0.0)
    var = (centered ** 2).sum(axis=0) / safe
    return var, cnt
