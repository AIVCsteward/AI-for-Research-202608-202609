"""Plan A — 蛋白级信噪比筛选。

用同板号技术重复估 σ²_noise、跨板号生物学重复估 σ²_total，按 SNR 筛出高信噪比
蛋白子集，重测子集上的跨板号 FC oracle。若 oracle 随子集缩小而上升，说明低 SNR
蛋白在拖累整体。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.oracle_common import (
    DEFAULT_K,
    cross_plate_oracle,
    load_cache,
    masked_var_across_pairs,
)

OUT = Path("experiments/outputs/oracle/plan_a_snr.json")


def main():
    c = load_cache()
    fc = c["fc_true"].astype(np.float64)
    fcm = c["fc_mask"]
    reps = c["replicate_pairs"]

    tech = reps[reps[:, 2] == 0]
    bio = reps[reps[:, 2] == 1]

    # 技术重复 → 测量噪音方差
    ti, tk = tech[:, 0], tech[:, 1]
    vt = fcm[ti] & fcm[tk]
    var_noise, _ = masked_var_across_pairs(
        np.where(vt, fc[ti] - fc[tk], 0.0), vt
    )

    # 生物学重复 → 总方差
    bi, bk = bio[:, 0], bio[:, 1]
    vb = fcm[bi] & fcm[bk]
    var_total, _ = masked_var_across_pairs(
        np.where(vb, fc[bi] - fc[bk], 0.0), vb
    )

    # SNR = 1 - noise/total，clip 到 [0,1]（负值说明该蛋白噪音>总方差，归 0）
    snr = 1.0 - np.divide(
        var_noise, var_total, out=np.ones_like(var_noise), where=var_total > 0
    )
    snr = np.clip(snr, 0.0, 1.0)

    order = np.argsort(-snr)  # 高 SNR 在前

    results = {
        "baseline_oracle": cross_plate_oracle(fc, fcm, reps, True),
        "snr_distribution": {
            "median": float(np.median(snr)),
            "p25": float(np.percentile(snr, 25)),
            "p75": float(np.percentile(snr, 75)),
            "frac_snr_gt_0.5": float((snr > 0.5).mean()),
        },
    }
    for K in DEFAULT_K:
        top = order[:K]
        results[f"K={K}"] = cross_plate_oracle(fc[:, top], fcm[:, top], reps, True)
        results[f"K={K}_median_snr"] = float(np.median(snr[top]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
