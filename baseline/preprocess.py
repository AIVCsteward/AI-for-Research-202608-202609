"""
数据预处理模块：对齐 → 过滤 → log2 → mask
严格按照设计文档 4.2 / 5.2 节流程实现
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_data():
    """加载原始数据，以 sample_ID 对齐元数据与蛋白质组矩阵"""
    # 训练/验证集
    meta_train = pd.read_csv(DATA_DIR / "WAYB_WAYC_metadata_train_val(1).csv")
    prot_train = pd.read_csv(DATA_DIR / "WAYB_WAYC_proteome_raw_train_val.csv")

    # 测试集
    meta_test = pd.read_csv(DATA_DIR / "WAYB_WAYC_metadata_test(1).csv")
    prot_test = pd.read_csv(DATA_DIR / "WAYB_WAYC_proteome_raw_test.csv")

    # 合并 train+val 和 test（统一预处理，后续按 split_final 拆分）
    meta = pd.concat([meta_train, meta_test], ignore_index=True)
    prot = pd.concat([prot_train, prot_test], ignore_index=True)

    print(f"元数据: {meta.shape}, 蛋白质组: {prot.shape}")
    print(f"split_final 分布:\n{meta['split_final'].value_counts().to_string()}")

    return meta, prot


def align_and_filter(meta, prot, missing_threshold=0.80):
    """
    以 sample_ID 对齐 → 仅用训练行计算缺失率 → 过滤高缺失蛋白

    返回:
        y_log2:        (N, kept_proteins) log2 转换后的蛋白质组数据
        mask_matrix:    (N, kept_proteins) 布尔 mask (True=有值)
        meta:           (N,) 对齐后的元数据
        kept_proteins:  保留的蛋白名列
    """
    # ---------- 1. 以 sample_ID 对齐 ----------
    meta = meta.set_index("sample_ID")
    prot = prot.set_index("sample_ID")

    # 确保 meta 和 prot 的样本完全对应
    common_ids = meta.index.intersection(prot.index)
    print(f"对齐后共同样本数: {len(common_ids)}")

    meta = meta.loc[common_ids]
    prot = prot.loc[common_ids]

    # ---------- 2. 仅用训练行计算缺失率 ----------
    train_mask = meta["split_final"] == "train"
    protein_cols = prot.columns  # 蛋白名列（第一列是 sample_ID，但已设为 index）
    missing_rate = prot.loc[train_mask, protein_cols].isna().mean(axis=0)
    kept = missing_rate < missing_threshold
    print(f"过滤: {len(protein_cols)} → {kept.sum()} 蛋白 (缺失率阈值 {missing_threshold})")

    prot_filtered = prot.loc[:, kept[kept].index]

    # ---------- 3. log2 转换（缺失位置保持 NaN）----------
    y_log2 = np.log2(prot_filtered.astype(float))

    # ---------- 4. 构建 mask 矩阵 ----------
    mask_matrix = ~y_log2.isna()

    print(f"预处理完成: y_log2={y_log2.shape}, "
          f"有效值比例={mask_matrix.values.mean():.1%}, "
          f"log2 值范围=[{y_log2.min().min():.1f}, {y_log2.max().max():.1f}]")

    return y_log2, mask_matrix, meta, kept[kept].index.tolist()


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
    is_control = meta["perturbation_no_concentration"].str.lower().isin(
        ["water", "dmso"]
    )
    return is_control


def identify_treatments(meta):
    """识别处理组样本（非 Water/DMSO 且非质控）"""
    is_control = identify_controls(meta)
    is_qc = meta["perturbation_no_concentration"].str.lower().str.contains(
        "quality|qc", na=False
    )
    return ~is_control & ~is_qc


if __name__ == "__main__":
    meta, prot = load_data()
    y_log2, mask, meta, proteins = align_and_filter(meta, prot)
    splits = get_split_masks(meta)

    print(f"\n各 split 样本数:")
    for k, v in splits.items():
        print(f"  {k}: {v.sum()}")

    print(f"\n对照组样本数: {identify_controls(meta).sum()}")
    print(f"处理组样本数: {identify_treatments(meta).sum()}")
