"""
================================================================================
AIVC Phase 0 + Phase 1: 完整 Baseline 流水线
================================================================================
Phase 0: 蛋白均值基线 + Matched Control 基线 → 端到端提交
Phase 1: 条件编码 MLP → mask-aware 训练 → 多方案对比

严格按照设计文档 3.2 节 / 4.2-4.3 节实现
================================================================================
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# 0. 路径与常量
# ============================================================================
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# 1. 数据加载与预处理
# ============================================================================
print("=" * 70)
print("Phase 0 | Step 1: 数据加载与预处理")
print("=" * 70)

# 加载
meta_train = pd.read_csv(DATA_DIR / "WAYB_WAYC_metadata_train_val(1).csv")
prot_train = pd.read_csv(DATA_DIR / "WAYB_WAYC_proteome_raw_train_val.csv")
meta_test = pd.read_csv(DATA_DIR / "WAYB_WAYC_metadata_test(1).csv")
prot_test = pd.read_csv(DATA_DIR / "WAYB_WAYC_proteome_raw_test.csv")

meta = pd.concat([meta_train, meta_test], ignore_index=True)
prot = pd.concat([prot_train, prot_test], ignore_index=True)
print(f"原始数据: meta={meta.shape}, prot={prot.shape}")

# --- 1a. 以 sample_ID 对齐 ---
meta = meta.set_index("sample_ID")
prot = prot.set_index("sample_ID")
common_ids = meta.index.intersection(prot.index)
meta = meta.loc[common_ids]
prot = prot.loc[common_ids]
print(f"对齐后样本数: {len(common_ids)}")

# --- 1b. 仅用训练行过滤高缺失蛋白 ---
train_mask = meta["split_final"] == "train"
protein_cols = prot.columns
missing_rate = prot.loc[train_mask, protein_cols].isna().mean(axis=0)
keep = missing_rate < 0.80
print(f"蛋白过滤: {len(protein_cols)} → {keep.sum()} (阈值 80% 缺失率)")

prot_filtered = prot.loc[:, keep[keep].index]
protein_names = keep[keep].index.tolist()
N_PROTEINS = len(protein_names)

# --- 1c. log2 转换 + mask 矩阵 ---
y_log2 = np.log2(prot_filtered.astype(float))
mask_matrix = ~y_log2.isna()  # True = 有观测值
print(f"log2 范围: [{y_log2.min().min():.1f}, {y_log2.max().max():.1f}]")
print(f"有效值比例: {mask_matrix.values.mean():.1%}")

# --- 1d. 识别对照组 ---
def is_control(meta_df):
    """识别 Water/DMSO 溶剂对照样本"""
    return meta_df["perturbation_no_concentration"].str.lower().isin(["water", "dmso"])

is_ctrl = is_control(meta)
print(f"对照组样本: {is_ctrl.sum()}, 处理组+质控: {(~is_ctrl).sum()}")

# --- 1e. 各 split mask ---
split_masks = {s: meta["split_final"] == s for s in meta["split_final"].unique()}
for k, v in split_masks.items():
    print(f"  {k}: {v.sum()} 样本")

# ============================================================================
# 2. Phase 0 | 基线一: 蛋白均值基线
# ============================================================================
print("\n" + "=" * 70)
print("Phase 0 | 基线一: 蛋白均值基线")
print("=" * 70)

# 训练集每个蛋白的 log2 均值（忽略 NaN）
protein_mean = y_log2.loc[train_mask].mean(axis=0)
print(f"蛋白均值向量: {protein_mean.shape}, 值范围 [{protein_mean.min():.1f}, {protein_mean.max():.1f}]")


def evaluate_global_r2(y_true, y_pred, mask):
    """
    Global R²: 把所有蛋白-样本对摊平计算
    使用 mask-aware 方式：只在有观测值的位置计算
    """
    y_t = y_true.fillna(0).values
    m = mask.values.astype(float)
    y_p = y_pred

    # 只算 mask=1 的位置
    ss_res = ((y_t - y_p) ** 2 * m).sum()
    # 全局均值（mask-aware）
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


# 在验证集上评估蛋白均值基线
val_splits = ["val_strain_only", "val_chem_only", "val_both", "val_time"]
test_splits = ["test_strain_only", "test_chem_only", "test_both", "test_time"]
all_eval_splits = val_splits + test_splits
print("\n蛋白均值基线 — 验证集+测试集评估:")
for split_name in all_eval_splits:
    mask = split_masks[split_name]
    if mask.sum() == 0:
        continue
    y_true = y_log2.loc[mask]
    m = mask_matrix.loc[mask]
    n_samples = mask.sum()
    pred = np.tile(protein_mean.values, (n_samples, 1))

    global_r2 = evaluate_global_r2(y_true, pred, m)
    pp_r2 = evaluate_per_protein_r2(y_true, pred, m)
    print(f"  {split_name:20s} | n={n_samples:4d} | Global R²={global_r2:.4f} | Per-Protein R²(median)={pp_r2:.4f}")

# ============================================================================
# 3. Phase 0 | 基线二: Matched Control 基线
# ============================================================================
print("\n" + "=" * 70)
print("Phase 0 | 基线二: Matched Control 基线")
print("=" * 70)

# 匹配键：data_source + instrument + Yeast_cell_plate + Strains +
#          Medium + Temperature + pert_time
MATCH_KEYS = ["data_source", "instrument", "Yeast_cell_plate",
              "Strains", "Medium", "Temperature", "pert_time"]

# 构建对照组索引：对每个匹配键组合，找到对应的 Water/DMSO 样本
control_meta = meta[is_ctrl].copy()
# 为对照组建立查找表：匹配键 → sample_ID 列表
control_lookup = {}
for sid, row in control_meta.iterrows():
    key = tuple(row[k] for k in MATCH_KEYS)
    control_lookup.setdefault(key, []).append(sid)

print(f"对照组唯一匹配键数: {len(control_lookup)}")


def matched_control_predict(target_meta, y_log2_df, fill_na_with_global=True):
    """
    对 target_meta 中的每个样本，查找 matched control 并返回其蛋白向量
    找不到匹配的样本：用全局 control 均值填充
    对照蛋白有 NA：用全局蛋白均值填充（对照样本本身也可能有缺失值）
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
                # 对照样本中也可能有 NA → 用全局蛋白均值填充
                na_mask = np.isnan(ctrl_vec)
                if na_mask.any():
                    ctrl_vec[na_mask] = protein_mean.values[na_mask]
                    na_filled_count += 1
                predictions.append(ctrl_vec)
                matched_count += 1
                continue
        # 回退：全局对照组均值（同样可能含 NA，需填充）
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


# 全局对照组均值（作为匹配失败时的回退）
control_mean = y_log2.loc[is_ctrl].mean(axis=0)

print("\nMatched Control 基线 — 验证集+测试集评估:")
for split_name in all_eval_splits:
    mask = split_masks[split_name]
    if mask.sum() == 0:
        continue
    target_meta_subset = meta.loc[mask]
    y_true = y_log2.loc[mask]
    m = mask_matrix.loc[mask]

    pred = matched_control_predict(target_meta_subset, y_log2)

    global_r2 = evaluate_global_r2(y_true, pred, m)
    pp_r2 = evaluate_per_protein_r2(y_true, pred, m)
    print(f"  {split_name:20s} | n={mask.sum():4d} | Global R²={global_r2:.4f} | Per-Protein R²(median)={pp_r2:.4f}")


# ============================================================================
# 4. Phase 1 | 条件编码 MLP
# ============================================================================
print("\n" + "=" * 70)
print("Phase 1 | 条件编码 MLP")
print("=" * 70)

import torch
import torch.nn as nn

torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {DEVICE}")


# --- 4a. 特征工程 ---
def build_condition_features(meta_df, fit_encoders=False, encoders=None):
    """
    将元数据编码为数值特征矩阵 (N, dim_in)
    包含: one-hot 编码 + 时间 cyclic 编码
    """
    features = []

    if encoders is None:
        encoders = {}
        fit_encoders = True

    # 类别特征 → one-hot
    cat_cols = {
        "strains": "Strains",
        "chemicals": "perturbation_no_concentration",
        "media": "Medium",
        "instruments": "instrument",
    }

    for name, col in cat_cols.items():
        if fit_encoders:
            encoders[name] = {v: i for i, v in enumerate(sorted(meta_df[col].unique()))}
        idx = meta_df[col].map(encoders[name]).fillna(0).astype(int).values
        n_cats = len(encoders[name])
        oh = np.eye(n_cats)[idx]
        features.append(oh)

    # 温度 → 二值
    temp = (meta_df["Temperature"].values == 37).astype(float).reshape(-1, 1)
    features.append(temp)

    # 时间 → cyclic sin/cos 编码
    time_hours = meta_df["pert_time"].astype(float).values
    # 先统一到小时单位（数据中有 min 和 h 两种单位，但 pert_time 值似乎已统一）
    # 实际数据中 pert_time 值有 15, 30, 60, 90, 120, 180, 240, 360, 720, 1440 等
    time_norm = 2 * np.pi * time_hours / time_hours.max()
    features.append(np.sin(time_norm).reshape(-1, 1))
    features.append(np.cos(time_norm).reshape(-1, 1))

    X = np.concatenate(features, axis=1).astype(np.float32)

    if fit_encoders:
        return X, encoders
    return X


X_all, encoders = build_condition_features(meta)
DIM_IN = X_all.shape[1]
print(f"条件特征维度: {DIM_IN}")
print(f"  Strains: {len(encoders['strains'])} 类")
print(f"  Chemicals: {len(encoders['chemicals'])} 类")
print(f"  Media: {len(encoders['media'])} 类")
print(f"  Instruments: {len(encoders['instruments'])} 类")


# --- 4b. 模型定义 ---
class ConditionMLP(nn.Module):
    """条件编码 MLP: one-hot → 隐层 → N_proteins 输出"""

    def __init__(self, dim_in, dim_out, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim_out),
        )

    def forward(self, x):
        return self.net(x)


model = ConditionMLP(DIM_IN, N_PROTEINS, hidden=256, dropout=0.1).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {n_params:,}")

# --- 4c. mask-aware 训练 ---
# 训练数据
X_train_t = torch.tensor(X_all[train_mask.values], dtype=torch.float32).to(DEVICE)
y_train_t = torch.tensor(
    y_log2.loc[train_mask].fillna(0).values, dtype=torch.float32
).to(DEVICE)
mask_train_t = torch.tensor(
    mask_matrix.loc[train_mask].values, dtype=torch.float32
).to(DEVICE)

# 验证数据（各 split）
val_data = {}
for split_name in val_splits:
    m = split_masks[split_name]
    if m.sum() == 0:
        continue
    val_data[split_name] = {
        "X": torch.tensor(X_all[m.values], dtype=torch.float32).to(DEVICE),
        "y_true": y_log2.loc[m],
        "mask": mask_matrix.loc[m],
        "y_filled": torch.tensor(
            y_log2.loc[m].fillna(0).values, dtype=torch.float32
        ).to(DEVICE),
        "mask_t": torch.tensor(
            mask_matrix.loc[m].values, dtype=torch.float32
        ).to(DEVICE),
    }


def mask_aware_mse(pred, y_filled, mask):
    """mask-aware MSE: 缺失值不贡献梯度"""
    diff = (pred - y_filled) ** 2
    return (diff * mask).sum() / mask.sum()


print("\n开始训练 MLP...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)

EPOCHS = 100
BATCH_SIZE = 256
N_TRAIN = X_train_t.shape[0]

best_val_loss = float("inf")
best_state = None
history = {"train_loss": [], "val_loss": {sn: [] for sn in val_data}}

for epoch in range(EPOCHS):
    model.train()
    # Mini-batch 训练
    perm = torch.randperm(N_TRAIN)
    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, N_TRAIN, BATCH_SIZE):
        idx = perm[i : i + BATCH_SIZE]
        optimizer.zero_grad()
        pred = model(X_train_t[idx])
        loss = mask_aware_mse(pred, y_train_t[idx], mask_train_t[idx])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    avg_train_loss = epoch_loss / n_batches
    history["train_loss"].append(avg_train_loss)

    # 验证
    model.eval()
    val_losses = []
    with torch.no_grad():
        for sn, vd in val_data.items():
            pred = model(vd["X"])
            v_loss = mask_aware_mse(pred, vd["y_filled"], vd["mask_t"]).item()
            history["val_loss"][sn].append(v_loss)
            val_losses.append(v_loss)

    avg_val_loss = np.mean(val_losses)
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
              f"train loss={avg_train_loss:.4f} | "
              f"val loss={avg_val_loss:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

# 加载最佳模型
model.load_state_dict(best_state)
print(f"训练完成, best val loss={best_val_loss:.4f}")

# ============================================================================
# 5. 评估与多方案对比
# ============================================================================
print("\n" + "=" * 70)
print("多方案对比")
print("=" * 70)


def mlp_predict(meta_subset):
    """用训练好的 MLP 对给定样本集做预测"""
    X = torch.tensor(
        build_condition_features(meta_subset, encoders=encoders), dtype=torch.float32
    ).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(X).cpu().numpy()
    return pred


# 各方案在验证集+测试集上的对比
print(f"\n{'Split':<20s} {'方法':<18s} {'Global R²':>10s} {'Per-Protein R²':>16s}")
print("-" * 68)

for split_name in all_eval_splits:
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
    pred_mc = matched_control_predict(meta_sub, y_log2)
    gr2_mc = evaluate_global_r2(y_true, pred_mc, mask)
    ppr2_mc = evaluate_per_protein_r2(y_true, pred_mc, mask)

    # MLP
    pred_mlp = mlp_predict(meta_sub)
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
# 6. 生成提交文件
# ============================================================================
print("\n" + "=" * 70)
print("生成 prediction.csv")
print("=" * 70)

# 测试集
test_mask = meta["split_final"].str.startswith("test")
test_meta = meta.loc[test_mask]
print(f"测试样本数: {test_mask.sum()}")

# MLP 预测
pred_test = mlp_predict(test_meta)

# Clamp 到合理 log2 范围（防止个别蛋白预测异常值）
# log2 intensity 合理范围约 [8, 36]
pred_test = np.clip(pred_test, 5.0, 40.0)
print(f"预测值范围 (clamp后): [{pred_test.min():.2f}, {pred_test.max():.2f}]")

# 注: control 偏移校准 (文档 5.6.3) 留到提分阶段
# 当前 MLP 仅在训练集上学到条件映射，校准可能引入训练-测试分布偏移

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
print(f"校验通过: 无 NA, 无 Inf")
print(f"提交文件: {submission.shape[0]} 样本 × {submission.shape[1]} 蛋白")
print(
    f"值范围: [{submission.values.min():.2f}, {submission.values.max():.2f}] (log2)"
)

# 保存
output_path = OUTPUT_DIR / "prediction.csv"
submission.to_csv(output_path)
print(f"\n已保存到: {output_path}")

# ============================================================================
# 7. 指标对齐诊断
# ============================================================================
print("\n" + "=" * 70)
print("基线对齐诊断")
print("=" * 70)

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

print("Phase 0+1 完成!")
