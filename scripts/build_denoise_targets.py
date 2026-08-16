"""构造低秩去噪 FC 目标（多 k 版本）与高 SNR 蛋白掩码。

一次 SVD，用多个 k 分别重构，保存 denoised_fc_target_k{k}.npy（训练样本顺序）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.oracle_common import load_cache, masked_var_across_pairs

OUT_DIR = Path("experiments/outputs/oracle")
K_LIST = [10, 20, 25, 30]
TOP_K_SNR = 500


def main():
    c = load_cache()
    fc = c["fc_true"].astype(np.float64)
    fcm = c["fc_mask"]
    reps = c["replicate_pairs"]
    treat_index = c["treat_index"]
    protein_names = c["protein_names"]
    meta = c["meta"]

    # ---- 低秩去噪（一次 SVD，多 k 重构）----
    fc_filled = np.nan_to_num(fc, nan=0.0)
    col_cnt = fcm.sum(axis=0).astype(np.float64)
    col_mean = np.divide(
        np.where(fcm, fc, 0.0).sum(axis=0), col_cnt,
        out=np.zeros(fc.shape[1]), where=col_cnt > 0,
    )
    centered = fc_filled - col_mean[None, :]
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    print(f"SVD 完成，奇异值 top10 = {np.round(s[:10], 1)}")

    # 训练样本顺序
    train_mask = meta["split_final"].astype(str).eq("train")
    train_sids = list(meta.index[train_mask])
    n_train = len(train_sids)
    n_prot = fc.shape[1]
    pos = {sid: i for i, sid in enumerate(train_sids)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for k in K_LIST:
        recon = (U[:, :k] * s[:k]) @ Vt[:k, :] + col_mean[None, :]
        recon = np.where(fcm, recon, np.nan).astype(np.float32)
        target = np.full((n_train, n_prot), np.nan, dtype=np.float32)
        for i, sid in enumerate(treat_index):
            if sid in pos:
                target[pos[sid]] = recon[i]
        np.save(OUT_DIR / f"denoised_fc_target_k{k}.npy", target)
        print(f"  k={k}: 已保存 denoised_fc_target_k{k}.npy")

    # ---- 高 SNR 掩码（诊断用，训练已证明子集掩码无效）----
    tech = reps[reps[:, 2] == 0]
    bio = reps[reps[:, 2] == 1]
    ti, tk = tech[:, 0], tech[:, 1]
    vt = fcm[ti] & fcm[tk]
    var_noise, _ = masked_var_across_pairs(np.where(vt, fc[ti] - fc[tk], 0.0), vt)
    bi, bk = bio[:, 0], bio[:, 1]
    vb = fcm[bi] & fcm[bk]
    var_total, _ = masked_var_across_pairs(np.where(vb, fc[bi] - fc[bk], 0.0), vb)
    snr = 1.0 - np.divide(var_noise, var_total, out=np.ones_like(var_noise), where=var_total > 0)
    snr = np.clip(snr, 0.0, 1.0)
    np.save(OUT_DIR / "snr_values.npy", snr.astype(np.float32))
    np.save(OUT_DIR / "denoise_protein_names.npy", np.asarray(protein_names))
    print(f"SNR 已保存（median={np.median(snr):.3f}，仅诊断）")


if __name__ == "__main__":
    main()
