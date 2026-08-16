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

# 溶剂对照（Water / DMSO）
CONTROL_NAMES = {"water", "dmso"}

# 评估 split 顺序
VAL_SPLITS = ["val_strain_only", "val_chem_only", "val_both", "val_time"]
TEST_SPLITS = ["test_strain_only", "test_chem_only", "test_both", "test_time"]


# ============================================================================
# 指标函数
# ============================================================================

# ============================================================================
# Phase 0 基线一：蛋白均值基线
# ============================================================================

def compute_protein_mean(y_log2, train_mask):
    """计算训练集每个蛋白的 log2 均值（忽略 NaN）"""
    pm = y_log2.loc[train_mask].mean(axis=0)
    print(f"蛋白均值向量: {pm.shape}, 值范围 [{pm.min():.1f}, {pm.max():.1f}]")
    return pm


def evaluate_protein_mean_baseline(y_log2, mask_matrix, protein_mean, split_masks):
    """在所有 val+test split 上评估蛋白均值基线（逐样本 corr）。"""
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

        per_corr = per_sample_corr(y_true, pred, mask)
        print(f"  {split_name:20s} | n={n_samples:4d} | per-sample corr={per_corr:.4f}")
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

        per_corr = per_sample_corr(y_true, pred, mask)
        print(f"  {split_name:20s} | n={m.sum():4d} | per-sample corr={per_corr:.4f}")
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
    """三方案（Protein Mean / Matched Control / MLP）全场景对比（逐样本 corr）。"""
    all_splits = VAL_SPLITS + TEST_SPLITS
    print(f"\n{'Split':<20s} {'方法':<18s} {'per-sample corr':>15s}")
    print("-" * 58)

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
        corr_pm = per_sample_corr(y_true, pred_pm, mask)

        # Matched Control
        pred_mc = matched_control_predict(
            meta_sub, y_log2, control_lookup, protein_mean, control_mean
        )
        corr_mc = per_sample_corr(y_true, pred_mc, mask)

        # MLP
        pred_mlp = mlp_predict(model, meta_sub, encoders, build_features_fn, device)
        corr_mlp = per_sample_corr(y_true, pred_mlp, mask)

        for method, corr in [
            ("Protein Mean", corr_pm),
            ("Matched Control", corr_mc),
            ("MLP (ours)", corr_mlp),
        ]:
            prefix = split_name if method == "Protein Mean" else ""
            print(f"{prefix:<20s} {method:<18s} {corr:>15.4f}")
        print("-" * 58)


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


# ============================================================================
# Person C: Fold Change 标签构造（数据纪律：仅从 train 计算）
# ============================================================================

def _resolve_train_mask(meta, train_mask=None):
    """返回与 meta index 对齐的 train-only boolean mask。"""
    if train_mask is None:
        if "split_final" not in meta.columns:
            raise ValueError("train_mask is required when split_final is unavailable")
        train_mask = meta["split_final"].astype(str).eq("train")
    elif isinstance(train_mask, pd.Series):
        train_mask = train_mask.reindex(meta.index)
        if train_mask.isna().any():
            raise ValueError("train_mask is not aligned with metadata index")
    else:
        train_mask = pd.Series(train_mask, index=meta.index)
    return train_mask.astype(bool)


def build_matched_control_pairs(meta, train_mask=None, match_keys=MATCH_KEYS):
    """构建精确 treatment-control 配对（仅使用训练集行）。

    只有当每个匹配键上都有 Water/DMSO 对照存在时，才保留 treatment。
    不使用全局对照回退，避免将估计基线变成伪造的 FC 标签。

    返回 DataFrame，每行一个 matched treatment，包含所有匹配的对照样本 ID 列表。
    """
    if not meta.index.is_unique:
        raise ValueError("metadata index must contain unique sample_ID values")
    missing_columns = [column for column in match_keys if column not in meta.columns]
    if missing_columns:
        raise KeyError(f"Missing matched-control columns: {missing_columns}")
    if "perturbation_no_concentration" not in meta.columns:
        raise KeyError("Missing perturbation_no_concentration column")

    train_only = _resolve_train_mask(meta, train_mask)
    perturbation = meta["perturbation_no_concentration"].astype(str).str.strip().str.lower()
    is_control = perturbation.isin(["water", "dmso"])
    is_qc = perturbation.str.contains("quality|qc", regex=True, na=False)

    control_lookup = {}
    for sample_id, row in meta.loc[train_only & is_control].iterrows():
        key = tuple(row[column] for column in match_keys)
        control_lookup.setdefault(key, []).append(sample_id)

    records = []
    for treatment_id, row in meta.loc[train_only & ~is_control & ~is_qc].iterrows():
        key = tuple(row[column] for column in match_keys)
        control_ids = tuple(control_lookup.get(key, ()))
        if not control_ids:
            continue
        records.append(
            {
                "treatment_sample_ID": treatment_id,
                "control_sample_ids": control_ids,
                "n_controls": len(control_ids),
            }
        )

    columns = ["treatment_sample_ID", "control_sample_ids", "n_controls"]
    return pd.DataFrame.from_records(records, columns=columns)


def compute_fold_change(
    meta,
    y_log2,
    mask_matrix,
    train_mask=None,
    pairs=None,
    match_keys=MATCH_KEYS,
):
    """计算仅训练集的 treatment-minus-matched-control FC 目标。

    多个匹配对照按蛋白逐列平均（仅使用有观测值的对照）。
    只有当 treatment 和至少一个匹配对照都有观测值时，该蛋白才进入 ``fc_mask``。

    返回字典：``pairs``, ``fc_true``, ``fc_mask``。
    ``fc_true`` 在无效位置填 NaN，方便检测误用。
    """
    if not y_log2.index.is_unique or not mask_matrix.index.is_unique:
        raise ValueError("target and mask indices must contain unique sample_ID values")
    if not y_log2.columns.equals(mask_matrix.columns):
        raise ValueError("y_log2 and mask_matrix protein columns must match exactly")
    missing_target_ids = meta.index.difference(y_log2.index)
    missing_mask_ids = meta.index.difference(mask_matrix.index)
    if len(missing_target_ids) or len(missing_mask_ids):
        raise ValueError("metadata, targets and mask must contain the same sample_ID values")

    train_only = _resolve_train_mask(meta, train_mask)
    if pairs is None:
        pairs = build_matched_control_pairs(meta, train_only, match_keys)
    else:
        pairs = pairs.copy()

    treatment_ids = pairs["treatment_sample_ID"].tolist()
    fc_values = np.full((len(treatment_ids), y_log2.shape[1]), np.nan, dtype=np.float32)
    fc_mask = np.zeros_like(fc_values, dtype=bool)

    aligned_y = y_log2.loc[meta.index].to_numpy(dtype=np.float32, copy=False)
    aligned_mask = mask_matrix.loc[meta.index].to_numpy(dtype=bool, copy=False)
    position_by_id = {sample_id: position for position, sample_id in enumerate(meta.index)}
    train_values = train_only.to_numpy(dtype=bool)

    treatment_positions = [position_by_id.get(sample_id) for sample_id in treatment_ids]
    if any(position is None for position in treatment_positions):
        raise ValueError("A treatment in pairs is missing from metadata")
    if any(not train_values[position] for position in treatment_positions):
        raise ValueError("Non-training treatment found in pairs")

    all_control_ids = {
        control_id
        for control_ids in pairs["control_sample_ids"]
        for control_id in control_ids
    }
    control_positions_by_id = {
        control_id: position_by_id.get(control_id) for control_id in all_control_ids
    }
    if any(position is None for position in control_positions_by_id.values()):
        raise ValueError("A control in pairs is missing from metadata")
    if any(not train_values[position] for position in control_positions_by_id.values()):
        raise ValueError("Non-training control found in pairs")

    pairs_with_position = pairs.copy()
    pairs_with_position["_output_position"] = np.arange(len(pairs_with_position))
    for control_id_tuple, group in pairs_with_position.groupby("control_sample_ids", sort=False):
        control_positions = [control_positions_by_id[control_id] for control_id in control_id_tuple]
        group_treatment_positions = [
            position_by_id[sample_id] for sample_id in group["treatment_sample_ID"]
        ]
        output_positions = group["_output_position"].to_numpy(dtype=int)

        controls = aligned_y[control_positions]
        controls_observed = aligned_mask[control_positions] & np.isfinite(controls)
        control_counts = controls_observed.sum(axis=0)
        control_sums = np.where(controls_observed, controls, 0.0).sum(axis=0)
        control_mean = np.divide(
            control_sums,
            control_counts,
            out=np.full(y_log2.shape[1], np.nan, dtype=np.float32),
            where=control_counts > 0,
        )

        treatments = aligned_y[group_treatment_positions]
        treatments_observed = aligned_mask[group_treatment_positions] & np.isfinite(treatments)
        valid = (
            treatments_observed
            & (control_counts > 0)[None, :]
            & np.isfinite(control_mean)[None, :]
        )
        group_fc = treatments - control_mean[None, :]
        fc_values[output_positions] = np.where(valid, group_fc, np.nan)
        fc_mask[output_positions] = valid

    fc_true_df = pd.DataFrame(fc_values, index=treatment_ids, columns=y_log2.columns)
    fc_true_df.index.name = "sample_ID"
    fc_mask_df = pd.DataFrame(fc_mask, index=treatment_ids, columns=y_log2.columns)
    fc_mask_df.index.name = "sample_ID"
    return {"pairs": pairs, "fc_true": fc_true_df, "fc_mask": fc_mask_df}


# ============================================================================
# 官方评测指标（GOAI 虚拟细胞统一开放榜六模块）
# ============================================================================
# 1. 绝对保真度（20%）——逐样本 corr/R²
# 2. 匹配对照原始 FC（25%）——PCC(Δ_pred, Δ_true)
# 3. 上下文均值残差（20%）——新化合物特异响应
# 4. 药物均值残差（20%）——新菌株背景调制
# 5. 双重未知 / 时间插值（10%）——由 1+2 组合
# 6. 高效应蛋白 DEP 检出（5%）——方向准确率 / 高效应 PCC
#
# 这些取代了旧封闭榜的「摊平 Global R²」和「逐蛋白 R² 中位数」两个自定诊断指标。

def pcc(a, b):
    """两个等长一维数组的 Pearson 相关；忽略非有限值，未定义返回 NaN。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 2:
        return float("nan")
    a = a[valid]
    b = b[valid]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return float((a * b).sum() / denom)


def per_sample_corr(y_true, y_pred, mask, min_obs=3):
    """绝对保真度：逐样本 Pearson 相关，返回跨样本均值。

    ``y_true``/``mask`` 为 DataFrame（样本×蛋白，含 NaN），``y_pred`` 为
    等形 ndarray。每个样本只在 ≥``min_obs`` 个观测蛋白上计算。
    """
    y_t = y_true.to_numpy(dtype=np.float64)
    m = mask.to_numpy(dtype=bool)
    y_p = np.asarray(y_pred, dtype=np.float64)
    corrs = []
    for i in range(y_t.shape[0]):
        valid = m[i]
        if int(valid.sum()) < min_obs:
            continue
        c = pcc(y_t[i][valid], y_p[i][valid])
        if np.isfinite(c):
            corrs.append(c)
    return float(np.mean(corrs)) if corrs else float("nan")


def per_protein_r2(y_true, y_pred, mask, min_obs=3):
    """逐蛋白 R² 中位数（诊断指标，非官方评分模块）。

    与 ``per_sample_corr`` 正交：这里沿「蛋白」轴看单个蛋白在样本间的预测
    准确度，用于暴露「模型是否只对高丰度蛋白准」。官方逐样本 corr 和高效应
    DEP 都看不到这个信息。只在 ≥``min_obs`` 个观测的蛋白上计算。
    """
    y_t = y_true.to_numpy(dtype=np.float64)
    m = mask.to_numpy(dtype=bool)
    y_p = np.asarray(y_pred, dtype=np.float64)
    scores = []
    for j in range(y_t.shape[1]):
        valid = m[:, j]
        if int(valid.sum()) < min_obs:
            continue
        obs = y_t[valid, j]
        ss_tot = ((obs - obs.mean()) ** 2).sum()
        if ss_tot > 1e-12:
            ss_res = ((obs - y_p[valid, j]) ** 2).sum()
            scores.append(1.0 - ss_res / ss_tot)
    return float(np.median(scores)) if scores else float("nan")


def masked_rmse(y_true, y_pred, mask):
    """Mask-aware RMSE（log2 单位）—— 只在观测位置计算，与训练 MSE 同口径。

    官方六模块不含 RMSE；此为诊断指标，用于对齐赛题解读基线表的
    「log2 RMSE」（蛋白均值 ≈0.87~1.0，matched control ≈0.38~0.43）。
    """
    y_t = y_true.to_numpy(dtype=np.float64)
    m = mask.to_numpy(dtype=bool)
    y_p = np.asarray(y_pred, dtype=np.float64)
    diff2 = (y_p - y_t) ** 2
    if int(m.sum()) == 0:
        return float("nan")
    return float(np.sqrt(diff2[m].mean()))


def fold_change_pcc(fc_pred, fc_true, fc_mask):
    """匹配对照原始 FC：PCC(Δ_pred, Δ_true)，在全部有效条目上摊平计算。"""
    return pcc(
        np.asarray(fc_pred, dtype=np.float64)[np.asarray(fc_mask, dtype=bool)],
        np.asarray(fc_true, dtype=np.float64)[np.asarray(fc_mask, dtype=bool)],
    )


def residual_pcc(fc_pred, fc_true, fc_mask, subtract):
    """残差 PCC：PCC(Δ_pred − μ, Δ_true − μ)，``subtract`` 为逐蛋白均值向量。"""
    valid = np.asarray(fc_mask, dtype=bool)
    sub = np.asarray(subtract, dtype=np.float64)
    return pcc(
        (np.asarray(fc_pred, dtype=np.float64) - sub)[valid],
        (np.asarray(fc_true, dtype=np.float64) - sub)[valid],
    )


def high_effect_direction_accuracy(fc_pred, fc_true, fc_mask, threshold=1.0):
    """高效应蛋白 DEP：对 |Δ_true|>threshold 的条目，方向一致的占比。"""
    valid = np.asarray(fc_mask, dtype=bool)
    fp = np.asarray(fc_pred, dtype=np.float64)
    ft = np.asarray(fc_true, dtype=np.float64)
    high = valid & np.isfinite(fp) & np.isfinite(ft) & (np.abs(ft) > threshold)
    if int(high.sum()) == 0:
        return float("nan")
    return float((np.sign(fp[high]) == np.sign(ft[high])).mean())


def compute_split_fold_change(meta, y_log2, mask_matrix, y_pred_df, split_mask, control_lookup):
    """为某一 split 的 treatment 样本计算 Δ_true 与 Δ_pred。

    ``control_lookup`` 应对全量数据构建（matched control 是真实测量，不限于 train）。
    ``y_pred_df`` 为 DataFrame（index=sample_ID）。返回 ``(fc_true, fc_pred, fc_mask,
    treat_meta)``，其中三个 FC 矩阵为 ndarray（treatment×蛋白），``treat_meta`` 为
    保留样本的元数据（用于查找 context/drug）。
    """
    sub = meta.loc[split_mask]
    pert = sub["perturbation_no_concentration"].astype(str).str.strip().str.lower()
    is_ctrl = pert.isin(CONTROL_NAMES)
    is_qc = pert.str.contains("quality|qc", regex=True, na=False)
    treat = sub[~is_ctrl & ~is_qc]
    if treat.empty:
        return None

    y_t = y_log2.to_numpy(dtype=np.float64)
    m_t = mask_matrix.to_numpy(dtype=bool)
    # y_pred_df 对齐到 meta.index（可能仅含 split 子集，用 reindex 补齐为 NaN）
    y_pred_full = y_pred_df.reindex(meta.index).to_numpy(dtype=np.float64)

    fc_true_rows, fc_pred_rows, fc_mask_rows = [], [], []
    kept = []
    for sid in treat.index:
        key = tuple(meta.loc[sid, k] for k in MATCH_KEYS)
        ctrl_ids = control_lookup.get(key, [])
        if not ctrl_ids:
            continue
        pos = meta.index.get_indexer(ctrl_ids)
        pos = pos[pos >= 0]
        if len(pos) == 0:
            continue

        ctrl_true = y_t[pos]
        ctrl_mask = m_t[pos]
        ctrl_count = ctrl_mask.sum(axis=0)
        ctrl_true_mean = np.divide(
            np.where(ctrl_mask, ctrl_true, 0.0).sum(axis=0),
            ctrl_count,
            out=np.full(y_t.shape[1], np.nan, dtype=np.float64),
            where=ctrl_count > 0,
        )
        ctrl_pred_mean = np.nanmean(y_pred_full[pos], axis=0)

        t_pos = meta.index.get_loc(sid)
        treat_true = y_t[t_pos]
        treat_mask = m_t[t_pos]
        treat_pred = y_pred_full[t_pos]

        fc_true_rows.append(treat_true - ctrl_true_mean)
        fc_pred_rows.append(treat_pred - ctrl_pred_mean)
        fc_mask_rows.append(treat_mask & (ctrl_count > 0) & np.isfinite(ctrl_true_mean))
        kept.append(sid)

    if not kept:
        return None
    return (
        np.vstack(fc_true_rows),
        np.vstack(fc_pred_rows),
        np.vstack(fc_mask_rows).astype(bool),
        meta.loc[kept],
    )


def build_train_residual_means(meta, y_log2, mask_matrix, train_mask):
    """从训练集构建上下文均值 μ_ctx 与药物均值 μ_drug（仅用 train）。

    返回 ``{"ctx_mean": {7元组: 蛋白向量}, "drug_mean": {药物: 蛋白向量}}``。
    均值按蛋白独立 mask-aware 计算（仅用有观测值的 treatment 累加）。
    """
    fc = compute_fold_change(meta, y_log2, mask_matrix, train_mask)
    fc_true = fc["fc_true"]
    fc_mask = fc["fc_mask"]
    v = fc_true.to_numpy(dtype=np.float64)
    m = fc_mask.to_numpy(dtype=bool)
    n_protein = v.shape[1]

    ctx_sum: dict = {}
    ctx_cnt: dict = {}
    drug_sum: dict = {}
    drug_cnt: dict = {}
    for i, sid in enumerate(fc_true.index):
        ctx = tuple(meta.loc[sid, k] for k in MATCH_KEYS)
        drug = str(meta.loc[sid, "perturbation_no_concentration"])
        row = np.where(m[i], v[i], 0.0)
        row_mask = m[i].astype(np.float64)
        ctx_sum[ctx] = ctx_sum.get(ctx, np.zeros(n_protein, dtype=np.float64)) + row
        ctx_cnt[ctx] = ctx_cnt.get(ctx, np.zeros(n_protein, dtype=np.float64)) + row_mask
        drug_sum[drug] = drug_sum.get(drug, np.zeros(n_protein, dtype=np.float64)) + row
        drug_cnt[drug] = drug_cnt.get(drug, np.zeros(n_protein, dtype=np.float64)) + row_mask

    ctx_mean = {
        key: np.divide(ctx_sum[key], ctx_cnt[key], out=np.full(n_protein, np.nan),
                       where=ctx_cnt[key] > 0)
        for key in ctx_sum
    }
    drug_mean = {
        key: np.divide(drug_sum[key], drug_cnt[key], out=np.full(n_protein, np.nan),
                       where=drug_cnt[key] > 0)
        for key in drug_sum
    }
    return {"ctx_mean": ctx_mean, "drug_mean": drug_mean}


def evaluate_official_metrics(
    meta, y_log2, mask_matrix, y_pred_df, split_mask, control_lookup, train_stats
):
    """为一个 split 计算全部官方指标，返回 dict。

    ``y_pred_df``：DataFrame（index=sample_ID）。
    """
    fc = compute_split_fold_change(meta, y_log2, mask_matrix, y_pred_df, split_mask, control_lookup)
    y_true_split = y_log2.loc[split_mask]
    mask_split = mask_matrix.loc[split_mask]
    pred_split = y_pred_df.reindex(meta.loc[split_mask].index).to_numpy(dtype=np.float64)
    metrics = {
        "n_samples": int(split_mask.sum()),
        "per_sample_corr": per_sample_corr(y_true_split, pred_split, mask_split),
        "per_protein_r2_median": per_protein_r2(y_true_split, pred_split, mask_split),
        "rmse": masked_rmse(y_true_split, pred_split, mask_split),
    }
    if fc is None:
        for key in ("fc_pcc", "context_residual_pcc", "drug_residual_pcc", "high_effect_dir_acc"):
            metrics[key] = float("nan")
        return metrics

    fc_true, fc_pred, fc_mask, treat_meta = fc
    metrics["fc_pcc"] = fold_change_pcc(fc_pred, fc_true, fc_mask)
    metrics["high_effect_dir_acc"] = high_effect_direction_accuracy(fc_pred, fc_true, fc_mask)

    # 上下文均值残差：逐 treatment 减去其 context 的训练药物均值
    ctx_vals_true, ctx_vals_pred, ctx_valid = [], [], []
    for i, sid in enumerate(treat_meta.index):
        key = tuple(meta.loc[sid, k] for k in MATCH_KEYS)
        mu = train_stats["ctx_mean"].get(key)
        if mu is None:
            continue
        ctx_vals_pred.append(fc_pred[i] - mu)
        ctx_vals_true.append(fc_true[i] - mu)
        ctx_valid.append(fc_mask[i])
    if ctx_valid:
        ctx_vals_pred = np.vstack(ctx_vals_pred)
        ctx_vals_true = np.vstack(ctx_vals_true)
        ctx_valid = np.vstack(ctx_valid).astype(bool)
        metrics["context_residual_pcc"] = fold_change_pcc(ctx_vals_pred, ctx_vals_true, ctx_valid)
    else:
        metrics["context_residual_pcc"] = float("nan")

    # 药物均值残差：逐 treatment 减去其药物的训练上下文均值
    drug_vals_true, drug_vals_pred, drug_valid = [], [], []
    for i, sid in enumerate(treat_meta.index):
        drug = meta.loc[sid, "perturbation_no_concentration"]
        mu = train_stats["drug_mean"].get(drug)
        if mu is None:
            continue
        drug_vals_pred.append(fc_pred[i] - mu)
        drug_vals_true.append(fc_true[i] - mu)
        drug_valid.append(fc_mask[i])
    if drug_valid:
        drug_vals_pred = np.vstack(drug_vals_pred)
        drug_vals_true = np.vstack(drug_vals_true)
        drug_valid = np.vstack(drug_valid).astype(bool)
        metrics["drug_residual_pcc"] = fold_change_pcc(drug_vals_pred, drug_vals_true, drug_valid)
    else:
        metrics["drug_residual_pcc"] = float("nan")

    return metrics
