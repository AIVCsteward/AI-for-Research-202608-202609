"""Plan B — 跨重复「共识」去噪标签。

对同条件 ≥3 重复的组合，用「其他样本的 mask-aware 中位数」作为该样本的去噪预测
（leave-one-out，避免含自身的虚假相关）。若去噪预测的 PCC 高于原始「单样本重复」
的 PCC，说明中位数平均确实压掉了样本级噪音，让真实效应更清晰。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from baseline.evaluation import pcc
from scripts.oracle_common import cross_plate_oracle, load_cache

OUT = Path("experiments/outputs/oracle/plan_b_consensus.json")


def _stat(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return None
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def main():
    c = load_cache()
    fc = c["fc_true"].astype(np.float64)
    fcm = c["fc_mask"]
    reps = c["replicate_pairs"]
    groups = c["group_to_samples"]

    baseline = cross_plate_oracle(fc, fcm, reps, True)

    # 原始跨板号单样本重复 PCC（与去噪做同口径对比）
    raw_cross = []
    denoised = []
    for i, j, is_cross in reps:
        if not is_cross:
            continue
        v = fcm[i] & fcm[j]
        if int(v.sum()) < 2:
            continue
        r = pcc(fc[i][v], fc[j][v])
        if np.isfinite(r):
            raw_cross.append(float(r))

    # leave-one-out 共识去噪
    for idxs in groups.values():
        if len(idxs) < 3:
            continue
        idxs = np.asarray(idxs)
        for i in idxs:
            others = idxs[idxs != i]
            med = np.nanmedian(fc[others], axis=0)
            valid = fcm[i] & np.isfinite(med)
            if int(valid.sum()) < 2:
                continue
            r = pcc(fc[i][valid], med[valid])
            if np.isfinite(r):
                denoised.append(float(r))

    results = {
        "baseline_oracle": baseline,
        "raw_single_replicate_cross_plate": _stat(raw_cross),
        "denoised_leave_one_out_median": _stat(denoised),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
