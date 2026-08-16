"""
数据加载与预处理模块：加载 → 对齐 → 过滤 → log2 → mask
严格按照设计文档 4.2 / 5.2 节流程实现

本模块是预处理功能的唯一权威来源。
baseline/preprocess.py 中的同名函数通过 re-export 保持向后兼容。
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_raw_data():
    """加载原始数据，以 sample_ID 对齐元数据与蛋白质组矩阵"""
    meta_train = pd.read_csv(DATA_DIR / "WAYB_WAYC_metadata_train_val(1).csv")
    prot_train = pd.read_csv(DATA_DIR / "WAYB_WAYC_proteome_raw_train_val.csv")

    meta_test = pd.read_csv(DATA_DIR / "WAYB_WAYC_metadata_test(1).csv")
    prot_test = pd.read_csv(DATA_DIR / "WAYB_WAYC_proteome_raw_test.csv")

    # 合并 train+val 和 test（统一预处理，后续按 split_final 拆分）
    meta = pd.concat([meta_train, meta_test], ignore_index=True)
    prot = pd.concat([prot_train, prot_test], ignore_index=True)

    print(f"原始数据: meta={meta.shape}, prot={prot.shape}")

    return meta, prot


def preprocess(meta, prot, missing_threshold=0.80):
    """
    以 sample_ID 对齐 → 仅用训练行计算缺失率 → 过滤高缺失蛋白
    → log2 转换 → 构建 mask 矩阵

    参数:
        meta: 元数据 DataFrame
        prot: 蛋白质组 DataFrame
        missing_threshold: 蛋白缺失率过滤阈值（默认 0.80）

    返回:
        y_log2:        (N, kept_proteins) log2 转换后的蛋白质组数据
        mask_matrix:    (N, kept_proteins) 布尔 mask (True=有值)
        meta:           (N,) 对齐后的元数据
        protein_names:  保留的蛋白名列
        train_mask:     (N,) 训练集 boolean mask
    """
    # ---------- 1. 以 sample_ID 对齐 ----------
    meta = meta.set_index("sample_ID")
    prot = prot.set_index("sample_ID")

    common_ids = meta.index.intersection(prot.index)
    meta = meta.loc[common_ids]
    prot = prot.loc[common_ids]
    print(f"对齐后样本数: {len(common_ids)}")

    # ---------- 2. 仅用训练行计算缺失率 ----------
    train_mask = meta["split_final"] == "train"
    protein_cols = prot.columns
    missing_rate = prot.loc[train_mask, protein_cols].isna().mean(axis=0)
    keep = missing_rate < missing_threshold
    print(f"蛋白过滤: {len(protein_cols)} → {keep.sum()} (阈值 {missing_threshold*100:.0f}% 缺失率)")

    prot_filtered = prot.loc[:, keep[keep].index]
    protein_names = keep[keep].index.tolist()

    # ---------- 3. log2 转换 ----------
    y_log2 = np.log2(prot_filtered.astype(float))

    # ---------- 4. 构建 mask 矩阵 ----------
    mask_matrix = ~y_log2.isna()
    print(f"log2 范围: [{y_log2.min().min():.1f}, {y_log2.max().max():.1f}]")
    print(f"有效值比例: {mask_matrix.values.mean():.1%}")

    return y_log2, mask_matrix, meta, protein_names, train_mask


def get_split_masks(meta):
    """返回各数据划分的 boolean mask"""
    splits = {}
    for split_name in meta["split_final"].unique():
        splits[split_name] = meta["split_final"] == split_name
    return splits


def identify_controls(meta):
    """
    识别对照组样本 (Water / DMSO)
    数据中 perturbation_no_concentration 列表示化学处理名称
    Water 和 DMSO 是溶剂对照
    """
    return meta["perturbation_no_concentration"].str.lower().isin(["water", "dmso"])


def identify_treatments(meta):
    """识别处理组样本（非 Water/DMSO 且非质控）"""
    is_control = identify_controls(meta)
    is_qc = meta["perturbation_no_concentration"].str.lower().str.contains(
        "quality|qc", na=False
    )
    return ~is_control & ~is_qc


# ---------------------------------------------------------------------------
# 向后兼容别名 — 供 baseline/preprocess.py 使用
# ---------------------------------------------------------------------------
align_and_filter = preprocess
load_data = load_raw_data
