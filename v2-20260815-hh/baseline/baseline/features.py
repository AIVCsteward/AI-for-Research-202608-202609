"""
特征工程模块：将元数据编码为数值特征矩阵

当前实现（Phase 1 baseline）:
  - 类别特征 → one-hot 编码
  - 温度 → 二值化
  - 时间 → sin/cos 循环编码

升级方向（设计文档 Phase 2+）:
  - Hash 编码替代 one-hot（高基数类别）
  - 菌株统计先验 PCA 锚定 (strain prior)
  - 化合物统计锚点 + 可学习残差 (chemical anchor-residual)
  - 交叉特征 (strain×medium, chemical×temperature)
"""
import numpy as np


def build_condition_features(meta_df, fit_encoders=False, encoders=None):
    """
    将元数据编码为数值特征矩阵 (N, dim_in)

    参数:
        meta_df:         元数据 DataFrame（index 为 sample_ID）
        fit_encoders:    是否从 meta_df 拟合编码器（首次调用时为 True）
        encoders:        已拟合的编码器字典（后续调用时传入）

    返回:
        X:               (N, dim_in) float32 特征矩阵
        encoders:        仅在 fit_encoders=True 时返回编码器字典
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
    # 数据纪律：周期归一化的 time_max 必须从训练集计算
    time_hours = meta_df["pert_time"].astype(float).values
    if fit_encoders:
        time_max = time_hours.max()
        encoders["_time_max"] = time_max
    else:
        time_max = encoders.get("_time_max", time_hours.max())
    time_norm = 2 * np.pi * time_hours / time_max
    features.append(np.sin(time_norm).reshape(-1, 1))
    features.append(np.cos(time_norm).reshape(-1, 1))

    X = np.concatenate(features, axis=1).astype(np.float32)

    if fit_encoders:
        return X, encoders
    return X
