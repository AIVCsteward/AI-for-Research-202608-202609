"""Plan D — 低秩分解分离「效应 vs 背景 vs 噪音」。

对 FC 矩阵（treatment×protein，NaN 填 0）做 SVD，用 top-k 奇异值重构作为「去噪后的
FC」，重测跨板号 oracle。若某个 k 下 oracle 上升，说明真实效应是低秩结构、噪音是
满秩的，可被 SVD 截断分离。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.oracle_common import cross_plate_oracle, load_cache

OUT = Path("experiments/outputs/oracle/plan_d_lowrank.json")
K_LIST = [10, 50, 100, 200, 400]


def main():
    c = load_cache()
    fc = c["fc_true"].astype(np.float64)
    fcm = c["fc_mask"]
    reps = c["replicate_pairs"]

    baseline = cross_plate_oracle(fc, fcm, reps, True)

    fc_filled = np.nan_to_num(fc, nan=0.0)
    # 去中心化（按蛋白均值，mask-aware），SVD 更稳定
    col_cnt = fcm.sum(axis=0).astype(np.float64)
    col_mean = np.divide(
        np.where(fcm, fc, 0.0).sum(axis=0), col_cnt,
        out=np.zeros(fc.shape[1]), where=col_cnt > 0,
    )
    centered = fc_filled - col_mean[None, :]

    U, s, Vt = np.linalg.svd(centered, full_matrices=False)

    results = {"baseline_oracle": baseline}
    for k in K_LIST:
        if k > len(s):
            break
        recon = (U[:, :k] * s[:k]) @ Vt[:k, :] + col_mean[None, :]
        results[f"k={k}"] = cross_plate_oracle(recon, fcm, reps, True)

    results["singular_values_top10"] = [float(x) for x in s[:10]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
