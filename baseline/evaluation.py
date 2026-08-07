"""
评估与提交模块：R² 指标、双基线评估、MLP 推理、结果对比、提交文件生成

升级方向（设计文档 Phase 2+）:
  - fold change Pearson 相关（FC PCC）指标
  - 分场景独立报告（不取均值）
  - 消融实验对比框架
"""
import numpy as np
import pandas as pd
import torch

# Matched Control 匹配键
MATCH_KEYS = [
    "data_source", "instrument", "Yeast_cell_plate",
    "Strains", "Medium", "Temperature", "pert_time",
]

# 评估 split 顺序
VAL_SPLITS = ["val_strain_only", "val_chem_only", "val_both", "val_time"]
TEST_SPLITS = ["test_strain_only", "test_chem_only", "test_both", "test_time"]


# ============================================================================
# 指标函数
# ============================================================================

def evaluate_global_r2(y_true, y_pred, mask):
    """
    Global R²: 把所有蛋白-样本对摊平计算（mask-aware）

    参数:
        y_true:  真实值 DataFrame/ndarray（含 NA）
        y_pred:  预测值 ndarray
        mask:    布尔 mask DataFrame（True=有观测值）
    """
    y_t = y_true.fillna(0).values
    m = mask.values.astype(float)
    y_p = y_pred

    ss_res = ((y_t - y_p) ** 2 * m).sum()
    grand_mean = (y_t * m).sum() / m.sum()
    ss_tot = ((y_t - grand_mean) ** 2 * m).sum()
    return 1.0 - ss_res / ss_tot


def evaluate_per_protein_r2(y_true, y_pred, mask):
    """
    逐蛋白 R²: 每个蛋白单独算 R²，返回中位数
    只在有 ≥3 个观测值的蛋白上计算
    """
    r2s = []
    y_t = y_true.values
    m = mask.values
    y_p = y_pred

    for j in range(y_t.shape[1]):
        valid = m[:, j] > 0
        if valid.sum() < 3:
            continue
        ss_res = ((y_t[valid, j] - y_p[valid, j]) ** 2).sum()
        ss_tot = ((y_t[valid, j] - y_t[valid, j].mean()) ** 2).sum()
        if ss_tot > 0:
            r2s.append(1.0 - ss_res / ss_tot)

    return np.median(r2s) if r2s else np.nan


# ============================================================================
# Phase 0 基线一：蛋白均值基线
# ============================================================================

def compute_protein_mean(y_log2, train_mask):
    """计算训练集每个蛋白的 log2 均值（忽略 NaN）"""
    pm = y_log2.loc[train_mask].mean(axis=0)
    print(f"蛋白均值向量: {pm.shape}, 值范围 [{pm.min():.1f}, {pm.max():.1f}]")
    return pm


def evaluate_protein_mean_baseline(y_log2, mask_matrix, protein_mean, split_masks):
    """在所有 val+test split 上评估蛋白均值基线"""
    all_splits = VAL_SPLITS + TEST_SPLITS
    print("\n蛋白均值基线 — 验证集+测试集评估:")
    for split_name in all_splits:
        m = split_masks[split_name]
        if m.sum() == 0:
            continue
        y_true = y_log2.loc[m]
        mask = mask_matrix.loc[m]
        n_samples = m.sum()
        pred = np.tile(protein_mean.values, (n_samples, 1))

        global_r2 = evaluate_global_r2(y_true, pred, mask)
        pp_r2 = evaluate_per_protein_r2(y_true, pred, mask)
        print(f"  {split_name:20s} | n={n_samples:4d} | "
              f"Global R²={global_r2:.4f} | Per-Protein R²(median)={pp_r2:.4f}")
    return True


# ============================================================================
# Phase 0 基线二：Matched Control 基线
# ============================================================================

def build_control_lookup(meta, y_log2, train_mask=None):
    """
    构建对照组匹配索引

    数据纪律：control_lookup 和 control_mean 仅从训练集构建，
    防止验证/测试集的对照样本信息泄露到基线预测中。

    参数:
        meta:        元数据 DataFrame（index 为 sample_ID）
        y_log2:      log2 蛋白表达矩阵
        train_mask:  训练集 boolean mask（None 则使用全部数据）

    返回:
        control_lookup:  {匹配键元组: [control_sample_ID, ...]}
        control_mean:    训练集对照组蛋白均值向量（匹配失败时的回退）
        is_ctrl:         全量数据对照组 boolean mask
    """
    # 识别对照组（全量数据，仅用于信息输出）
    is_ctrl = meta["perturbation_no_concentration"].str.lower().isin(["water", "dmso"])
    print(f"对照组样本(全量): {is_ctrl.sum()}, 处理组+质控: {(~is_ctrl).sum()}")

    # 确定构建 lookup 和 mean 的数据范围
    if train_mask is not None:
        work_mask = train_mask & is_ctrl
        print(f"对照组样本(仅train): {work_mask.sum()}  ← control_lookup & control_mean 从此计算")
    else:
        work_mask = is_ctrl

    control_meta = meta[work_mask].copy()
    control_lookup = {}
    for sid, row in control_meta.iterrows():
        key = tuple(row[k] for k in MATCH_KEYS)
        control_lookup.setdefault(key, []).append(sid)

    print(f"对照组唯一匹配键数: {len(control_lookup)}")

    # control_mean 仅从训练集对照计算（防泄露）
    control_mean = y_log2.loc[work_mask].mean(axis=0)
    return control_lookup, control_mean, is_ctrl


def matched_control_predict(
    target_meta, y_log2_df, control_lookup, protein_mean, control_mean
):
    """
    对 target_meta 中的每个样本，查找 matched control 并返回其蛋白向量

    找不到匹配的样本 → 用全局 control 均值填充
    对照蛋白有 NA → 用全局蛋白均值填充
    """
    predictions = []
    matched_count = 0
    fallback_count = 0
    na_filled_count = 0

    for sid, row in target_meta.iterrows():
        key = tuple(row[k] for k in MATCH_KEYS)
        if key in control_lookup and len(control_lookup[key]) > 0:
            ctrl_sid = control_lookup[key][0]
            if ctrl_sid in y_log2_df.index:
                ctrl_vec = y_log2_df.loc[ctrl_sid].values.copy()
                na_mask = np.isnan(ctrl_vec)
                if na_mask.any():
                    ctrl_vec[na_mask] = protein_mean.values[na_mask]
                    na_filled_count += 1
                predictions.append(ctrl_vec)
                matched_count += 1
                continue
        # 回退
        ctrl_vec = control_mean.values.copy()
        na_mask = np.isnan(ctrl_vec)
        if na_mask.any():
            ctrl_vec[na_mask] = protein_mean.values[na_mask]
        predictions.append(ctrl_vec)
        fallback_count += 1

    if matched_count > 0 and na_filled_count > 0:
        print(f"  匹配: {matched_count}, 回退: {fallback_count}, "
              f"其中 {na_filled_count} 个对照样本的部分蛋白用了全局均值填充 NA")
    else:
        print(f"  匹配: {matched_count}, 回退: {fallback_count}")
    return np.array(predictions)


def evaluate_matched_control_baseline(
    meta, y_log2, mask_matrix, split_masks,
    control_lookup, protein_mean, control_mean
):
    """在所有 val+test split 上评估 Matched Control 基线"""
    all_splits = VAL_SPLITS + TEST_SPLITS
    print("\nMatched Control 基线 — 验证集+测试集评估:")
    for split_name in all_splits:
        m = split_masks[split_name]
        if m.sum() == 0:
            continue
        target_meta_subset = meta.loc[m]
        y_true = y_log2.loc[m]
        mask = mask_matrix.loc[m]

        pred = matched_control_predict(
            target_meta_subset, y_log2,
            control_lookup, protein_mean, control_mean
        )

        global_r2 = evaluate_global_r2(y_true, pred, mask)
        pp_r2 = evaluate_per_protein_r2(y_true, pred, mask)
        print(f"  {split_name:20s} | n={m.sum():4d} | "
              f"Global R²={global_r2:.4f} | Per-Protein R²(median)={pp_r2:.4f}")
    return True


# ============================================================================
# Phase 1 MLP：推理与对比评估
# ============================================================================

def mlp_predict(model, meta_subset, encoders, build_features_fn, device):
    """用训练好的 MLP 对给定样本集做预测"""
    X = torch.tensor(
        build_features_fn(meta_subset, encoders=encoders),
        dtype=torch.float32,
    ).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(X).cpu().numpy()
    return pred


def evaluate_all_splits(
    model, meta, y_log2, mask_matrix, split_masks,
    encoders, build_features_fn, device,
    protein_mean, control_lookup, control_mean,
):
    """三方案（Protein Mean / Matched Control / MLP）全场景对比"""
    all_splits = VAL_SPLITS + TEST_SPLITS
    print(f"\n{'Split':<20s} {'方法':<18s} {'Global R²':>10s} {'Per-Protein R²':>16s}")
    print("-" * 68)

    for split_name in all_splits:
        m = split_masks[split_name]
        if m.sum() == 0:
            continue
        y_true = y_log2.loc[m]
        mask = mask_matrix.loc[m]
        n = m.sum()
        meta_sub = meta.loc[m]

        # Protein Mean
        pred_pm = np.tile(protein_mean.values, (n, 1))
        gr2_pm = evaluate_global_r2(y_true, pred_pm, mask)
        ppr2_pm = evaluate_per_protein_r2(y_true, pred_pm, mask)

        # Matched Control
        pred_mc = matched_control_predict(
            meta_sub, y_log2, control_lookup, protein_mean, control_mean
        )
        gr2_mc = evaluate_global_r2(y_true, pred_mc, mask)
        ppr2_mc = evaluate_per_protein_r2(y_true, pred_mc, mask)

        # MLP
        pred_mlp = mlp_predict(model, meta_sub, encoders, build_features_fn, device)
        gr2_mlp = evaluate_global_r2(y_true, pred_mlp, mask)
        ppr2_mlp = evaluate_per_protein_r2(y_true, pred_mlp, mask)

        for method, gr2, ppr2 in [
            ("Protein Mean", gr2_pm, ppr2_pm),
            ("Matched Control", gr2_mc, ppr2_mc),
            ("MLP (ours)", gr2_mlp, ppr2_mlp),
        ]:
            prefix = split_name if method == "Protein Mean" else ""
            print(f"{prefix:<20s} {method:<18s} {gr2:>10.4f} {ppr2:>16.4f}")
        print("-" * 68)


# ============================================================================
# 提交文件生成
# ============================================================================

def generate_submission(
    model, meta, protein_names,
    encoders, build_features_fn,
    output_dir, device,
    clamp_min=5.0, clamp_max=40.0,
):
    """
    对全部 test 样本做 MLP 推理，生成 prediction.csv

    参数:
        model:              训练好的模型
        meta:               元数据
        protein_names:      蛋白名列
        encoders:           特征编码器
        build_features_fn:  特征构建函数
        output_dir:         输出目录
        device:             推理设备
        clamp_min, clamp_max: 预测值 clamp 范围
    """
    test_mask = meta["split_final"].str.startswith("test")
    test_meta = meta.loc[test_mask]
    print(f"测试样本数: {test_mask.sum()}")

    # MLP 预测
    pred_test = mlp_predict(model, test_meta, encoders, build_features_fn, device)

    # Clamp 到合理 log2 范围
    pred_test = np.clip(pred_test, clamp_min, clamp_max)
    print(f"预测值范围 (clamp后): [{pred_test.min():.2f}, {pred_test.max():.2f}]")

    # 构建 DataFrame
    submission = pd.DataFrame(
        pred_test,
        index=test_meta.index,
        columns=protein_names,
    )
    submission.index.name = "sample_ID"

    # 校验
    assert not submission.isna().any().any(), "存在 NA 值！"
    assert np.isfinite(submission.values).all(), "存在 Inf 值！"
    print("校验通过: 无 NA, 无 Inf")
    print(f"提交文件: {submission.shape[0]} 样本 × {submission.shape[1]} 蛋白")
    print(f"值范围: [{submission.values.min():.2f}, {submission.values.max():.2f}] (log2)")

    # 保存
    output_path = output_dir / "prediction.csv"
    submission.to_csv(output_path)
    print(f"\n已保存到: {output_path}")

    return submission


# ============================================================================
# 基线对齐诊断
# ============================================================================

def print_diagnostics():
    """打印基线对齐参考值"""
    print("""
赛题报告参考值 (来自设计文档 4.2.3):
  Protein Mean:   Global R² ≈ 0.87, Per-Protein R² ≈ -0.06
  Matched Control: Global R² ≈ 0.98, Per-Protein R² ≈ 0.72-0.84
  逐蛋白 R² 中位数 > 0 → MLP 有意义
  超过 Matched Control → MLP 学到了化合物扰动信息

关键认知:
  - Global R² 区分度极低（所有模型都接近天花板）
  - 真正的区分度在 Per-Protein R²、fold change、残差类指标
  - Matched Control 是「不建模」情况下的最强 baseline
""")
