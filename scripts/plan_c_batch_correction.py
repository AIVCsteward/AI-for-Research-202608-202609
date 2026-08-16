"""Plan C — 批次效应校正。

用各 plate 的 Water/DMSO 对照估计「蛋白特异性批次偏移」（= plate 对照均值 − 全局
对照均值），校正所有样本后重算 FC，重测跨板号 oracle。若 oracle 上升，说明批次
效应是压低 FC 可复现性的主因之一；若不变，说明 matched control 已抵消了它。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.oracle_common import PLATE_COLS, cross_plate_oracle, load_cache

OUT = Path("experiments/outputs/oracle/plan_c_batch.json")


def main():
    c = load_cache()
    y = c["y_log2"].astype(np.float64)
    mask = c["mask"]
    meta = c["meta"]
    protein_names = c["protein_names"]
    reps = c["replicate_pairs"]

    from baseline.evaluation import build_matched_control_pairs, compute_fold_change

    train_mask = meta["split_final"].astype(str).eq("train")
    pert = meta["perturbation_no_concentration"].astype(str).str.lower().str.strip()
    is_ctrl = pert.isin(["water", "dmso"])

    # 原始 FC oracle（baseline，用缓存里的 fc_true）
    baseline = cross_plate_oracle(c["fc_true"], c["fc_mask"], reps, True)

    # 全局对照均值（mask-aware）
    ctrl_idx = np.where(is_ctrl.values)[0]
    cnt = mask[ctrl_idx].sum(0)
    global_ctrl = np.divide(
        np.where(mask[ctrl_idx], y[ctrl_idx], 0.0).sum(0), cnt,
        out=np.full(y.shape[1], np.nan), where=cnt > 0,
    )

    # 每 plate 批次偏移
    plate_of = meta[PLATE_COLS].apply(tuple, axis=1)
    pos_of = {sid: i for i, sid in enumerate(meta.index)}
    batch_effect = np.zeros((len(meta), y.shape[1]), dtype=np.float64)
    n_plates = 0
    for plate, grp in meta.groupby(plate_of):
        idxs = grp.index
        pos = np.array([pos_of[sid] for sid in idxs])
        ctrl_labels = idxs[is_ctrl.loc[idxs].values]
        if len(ctrl_labels) == 0:
            continue
        ctrl_in = np.array([pos_of[sid] for sid in ctrl_labels])
        cnt_p = mask[ctrl_in].sum(0)
        plate_mean = np.divide(
            np.where(mask[ctrl_in], y[ctrl_in], 0.0).sum(0), cnt_p,
            out=np.full(y.shape[1], np.nan), where=cnt_p > 0,
        )
        eff = plate_mean - global_ctrl
        batch_effect[pos] = eff[None, :]
        n_plates += 1

    eff_valid = np.isfinite(batch_effect)
    y_corr = np.where(eff_valid, y - batch_effect, y)
    print(f"校正了 {n_plates} 个 plate 的批次偏移")

    # 重算 FC
    y_corr_df = pd.DataFrame(y_corr, index=meta.index, columns=protein_names)
    mask_df = pd.DataFrame(mask, index=meta.index, columns=protein_names)
    pairs = build_matched_control_pairs(meta, train_mask)
    fc = compute_fold_change(meta, y_corr_df, mask_df, train_mask, pairs)
    fc_corr = fc["fc_true"].to_numpy(dtype=np.float64)
    fcm_corr = fc["fc_mask"].to_numpy(dtype=bool)

    # 确认行序与缓存 treat_index 一致（compute_fold_change 是确定性的）
    assert list(fc["fc_true"].index) == list(c["treat_index"]), "treat 行序不一致"

    corrected = cross_plate_oracle(fc_corr, fcm_corr, reps, True)

    results = {
        "baseline_oracle": baseline,
        "batch_corrected_oracle": corrected,
        "n_plates_corrected": n_plates,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
