"""
================================================================================
AIVC Phase 0 + Phase 1: 完整 Baseline 流水线 (模块化重构版)
================================================================================
Phase 0: 蛋白均值基线 + Matched Control 基线 → 端到端提交
Phase 1: 条件编码 MLP → mask-aware 训练 → 多方案对比

严格按照设计文档 3.2 节 / 4.2-4.3 节实现

模块结构:
  baseline/data.py       — 数据加载 & 预处理
  baseline/features.py   — 特征工程
  baseline/model.py      — 模型定义 (ConditionMLP)
  aivc/training.py      — 训练循环 & 损失函数
  baseline/evaluation.py — 指标、基线评估、提交生成
================================================================================
"""
import numpy as np
import torch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from baseline.data import (
    load_raw_data, preprocess, get_split_masks, identify_controls,
)
from baseline.features import build_condition_features, fit_feature_encoders
from baseline.model import ConditionMLP
from aivc.training import prepare_training_data, train
from baseline.evaluation import (
    compute_protein_mean,
    evaluate_protein_mean_baseline,
    build_control_lookup,
    evaluate_matched_control_baseline,
    evaluate_all_splits,
    generate_submission,
    print_diagnostics,
    VAL_SPLITS, TEST_SPLITS,
)

# ============================================================================
# 0. 全局配置
# ============================================================================
RANDOM_SEED = 42
OUTPUT_DIR = Path(__file__).parent

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # ========================================================================
    # 1. 数据加载与预处理
    # ========================================================================
    print("=" * 70)
    print("Phase 0 | Step 1: 数据加载与预处理")
    print("=" * 70)

    meta, prot = load_raw_data()
    y_log2, mask_matrix, meta, protein_names, train_mask = preprocess(meta, prot)

    # 识别对照组 & 构建 split masks
    is_ctrl = identify_controls(meta)
    split_masks = get_split_masks(meta)
    for k, v in split_masks.items():
        print(f"  {k}: {v.sum()} 样本")

    # ========================================================================
    # 2. Phase 0 基线一：蛋白均值基线
    # ========================================================================
    print("\n" + "=" * 70)
    print("Phase 0 | 基线一: 蛋白均值基线")
    print("=" * 70)

    protein_mean = compute_protein_mean(y_log2, train_mask)
    evaluate_protein_mean_baseline(y_log2, mask_matrix, protein_mean, split_masks)

    # ========================================================================
    # 3. Phase 0 基线二：Matched Control 基线
    # ========================================================================
    print("\n" + "=" * 70)
    print("Phase 0 | 基线二: Matched Control 基线")
    print("=" * 70)

    control_lookup, control_mean, _ = build_control_lookup(meta, y_log2, train_mask)
    evaluate_matched_control_baseline(
        meta, y_log2, mask_matrix, split_masks,
        control_lookup, protein_mean, control_mean,
    )

    # ========================================================================
    # 4. Phase 1：条件编码 MLP
    # ========================================================================
    print("\n" + "=" * 70)
    print("Phase 1 | 条件编码 MLP")
    print("=" * 70)
    print(f"设备: {DEVICE}")

    # 4a. 特征工程
    # 数据纪律：类别映射、统计锚点、PCA 和最终 256 维投影均仅在 train 上拟合。
    encoders = fit_feature_encoders(
        meta.loc[train_mask],
        y_log2.loc[train_mask],
        mask_matrix.loc[train_mask],
    )
    # 用训练集拟合的编码器 transform 全部数据；unseen 类别使用统计 fallback、
    # all-zero categorical one-hot 和 deterministic hash，而不是映射到第一个已知类别。
    X_all = build_condition_features(meta, encoders=encoders)
    DIM_IN = X_all.shape[1]
    N_PROTEINS = len(protein_names)
    print(f"条件特征维度: {DIM_IN} (raw={encoders['raw_dim']})")
    for name in ["strains", "chemicals", "media", "instruments"]:
        print(f"  {name.capitalize()}: {len(encoders[name])} 类")

    # 4b. 模型定义
    model = ConditionMLP(DIM_IN, N_PROTEINS, hidden=256, dropout=0.1).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    # 4c. 准备数据 & 训练
    X_train_t, y_train_t, mask_train_t, val_data = prepare_training_data(
        X_all, y_log2, mask_matrix, train_mask, split_masks, VAL_SPLITS, DEVICE,
    )

    print("\n开始训练 MLP...")
    model, history = train(
        model, X_train_t, y_train_t, mask_train_t, val_data,
        epochs=100, batch_size=256, lr=1e-3, weight_decay=1e-5, device=DEVICE,
    )

    # ========================================================================
    # 5. 多方案对比
    # ========================================================================
    print("\n" + "=" * 70)
    print("多方案对比")
    print("=" * 70)

    evaluate_all_splits(
        model, meta, y_log2, mask_matrix, split_masks,
        encoders, build_condition_features, DEVICE,
        protein_mean, control_lookup, control_mean,
    )

    # ========================================================================
    # 6. 生成提交文件
    # ========================================================================
    print("\n" + "=" * 70)
    print("生成 prediction.csv")
    print("=" * 70)

    generate_submission(
        model, meta, protein_names,
        encoders, build_condition_features, OUTPUT_DIR, DEVICE,
    )

    # ========================================================================
    # 7. 基线对齐诊断
    # ========================================================================
    print("\n" + "=" * 70)
    print("基线对齐诊断")
    print("=" * 70)
    print_diagnostics()

    print("Phase 0+1 完成!")


if __name__ == "__main__":
    main()
