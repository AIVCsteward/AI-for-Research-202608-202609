"""
蛋白共表达图构建模块 (B4)

从训练集蛋白表达数据中计算 Pearson 相关矩阵 → KNN 图，
用作 GNN 的固定图结构（不参与梯度更新）。

数据纪律: 所有统计量仅从训练集计算。

算法: 向量化 mask-aware Pearson 相关计算（避免 O(P²N) 逐对循环），
      对 4422 蛋白 × 5920 样本规模可在几秒内完成。
"""
import numpy as np
import torch


def build_protein_graph(y_train_log2, mask_train, k=5, threshold=0.7):
    """
    从训练集蛋白表达数据构建 KNN 图

    参数:
        y_train_log2: (N_train, P) log2 蛋白表达 DataFrame
        mask_train:   (N_train, P) 观测 mask DataFrame
        k:            K 近邻数
        threshold:    |Pearson r| 低于此值的边被裁减

    返回:
        edge_index: torch.LongTensor (2, E)  PyG 格式边列表
    """
    P = y_train_log2.shape[1]
    y = y_train_log2.values.astype(np.float64)  # (N, P)
    m = mask_train.values.astype(bool)           # (N, P)

    # ── 1. 向量化 mask-aware Pearson 相关矩阵 ──
    # 数据纪律：均值、方差仅从各蛋白的有效观测样本计算
    y_filled = y.copy()
    y_filled[~m] = 0.0
    col_counts = m.sum(axis=0).astype(np.float64)  # (P,)
    col_means = y_filled.sum(axis=0) / np.maximum(col_counts, 1)

    # 中心化 + NA填0
    y_centered = np.where(m, y - col_means[np.newaxis, :], 0.0)

    # 协方差矩阵：(P, P)，仅使用两蛋白都有观测的样本对
    joint_counts = m.T.astype(np.float64) @ m.astype(np.float64)  # (P, P)
    cov = (y_centered.T @ y_centered) / np.maximum(joint_counts, 1.0)

    # 每蛋白标准差（mask-aware）
    y_sq = np.where(m, (y - col_means[np.newaxis, :]) ** 2, 0.0)
    col_vars = y_sq.sum(axis=0) / np.maximum(col_counts, 1)
    col_stds = np.sqrt(np.maximum(col_vars, 1e-20))

    # Pearson r = cov / (std_i * std_j)
    denom = np.outer(col_stds, col_stds)
    corr_matrix = cov / np.maximum(denom, 1e-20)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

    pair_count = (joint_counts >= 10).sum()
    print(f"蛋白相关矩阵: {P}x{P}, {pair_count} 对有足够共享样本 (>=10)")

    # ── 2. KNN + threshold 构建边 ──
    # 排除自环
    np.fill_diagonal(corr_matrix, -np.inf)

    edges = []
    for i in range(P):
        # 取 |corr| 最大的 k 个邻居
        top_k_indices = np.argpartition(-np.abs(corr_matrix[i]), k)[:k]
        for j in top_k_indices:
            r = corr_matrix[i, j]
            if abs(r) >= threshold:
                edges.append([i, j])

    if len(edges) == 0:
        print("[WARN] 无边生成！请降低 threshold 或增大 k")
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # ── 3. 图统计 ──
    degrees = np.bincount(edge_index[0].numpy(), minlength=P) if edge_index.shape[1] > 0 else np.zeros(P, dtype=int)
    isolated = (degrees == 0).sum()
    avg_deg = edge_index.shape[1] / P if P > 0 else 0
    nodes_in_graph = len(
        set(edge_index[0].tolist()) | set(edge_index[1].tolist())
    ) if edge_index.shape[1] > 0 else 0

    print(f"图统计: {P} 节点, {edge_index.shape[1]} 边, "
          f"平均度 {avg_deg:.1f}, {isolated} 孤立节点")
    print(f"图中覆盖蛋白: {nodes_in_graph}/{P}")

    return edge_index


# ══════════════════════════════════════════════════════════════════════════════
# 自检（依赖真实数据）
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from baseline.data import load_raw_data, preprocess  # shared data layer

    print("=" * 54)
    print("B4: build_protein_graph 自检")
    print("=" * 54)

    # 加载 + 预处理
    meta, prot = load_raw_data()
    y_log2, mask_matrix, meta, protein_names, train_mask = preprocess(meta, prot)

    N_PROTEINS = len(protein_names)
    print(f"\n蛋白数: {N_PROTEINS}")
    print(f"训练样本: {train_mask.sum()}")

    # 构建图
    edge_index = build_protein_graph(
        y_log2.loc[train_mask],
        mask_matrix.loc[train_mask],
        k=5,
        threshold=0.7,
    )

    # 检查
    assert edge_index.shape[0] == 2, f"edge_index shape[0] != 2: {edge_index.shape}"
    if edge_index.shape[1] > 0:
        assert edge_index.max() < N_PROTEINS, \
            f"边索引超出蛋白范围! max={edge_index.max()}, P={N_PROTEINS}"
        assert edge_index.min() >= 0
    print(f"\nedge_index shape: {edge_index.shape} [OK]")

    print("\nB4 [OK]")
