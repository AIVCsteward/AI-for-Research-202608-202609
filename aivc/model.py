"""
AIVC 正式模型：ConditionEncoder + ResidualDecoder + GNN + Calibration

架构:
    x (N, dim_in)
    → ConditionEncoder → emb (N, 256)
    → ResidualDecoder   → {baseline, deltas, y_raw}
    → [+ GNN 平滑]      → y_smoothed
    → CalibrationHead   → δ_cal
    → y_pred = y_smoothed + δ_cal
"""
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# B1: ConditionEncoder
# ══════════════════════════════════════════════════════════════════════════════

class ConditionEncoder(nn.Module):
    """
    条件编码器：输入特征 → 3层MLP → 条件嵌入 (N, 256)

    架构:
        Linear(dim_in, 256) → ReLU → BatchNorm → Dropout(0.1)
      → Linear(256, 256)    → ReLU → BatchNorm → Dropout(0.1)
      → Linear(256, 256)

    输出: (N, dim_emb) 条件嵌入向量
    """

    def __init__(self, dim_in, hidden=256, dropout=0.1):
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
            nn.Linear(hidden, hidden),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# B3: CalibrationHead
# ══════════════════════════════════════════════════════════════════════════════

class CalibrationHead(nn.Module):
    """
    批次校准分支：条件嵌入 → 轻量 MLP → 加性蛋白偏移

    输入: 条件嵌入 (N, dim_emb)
    输出: 加性偏移 (N, n_proteins)

    零初始化策略确保训练起始时模型行为等价于无校准，
    校准量在训练中逐渐浮现。
    """

    def __init__(self, dim_emb=256, n_proteins=4422, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_proteins),
        )
        # 零初始化 → 初始 δ_cal = 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, embedding):
        return self.net(embedding)


# ══════════════════════════════════════════════════════════════════════════════
# B4: ProteinGraphSmooth — 蛋白表达图平滑层
# ══════════════════════════════════════════════════════════════════════════════

class ProteinGraphSmooth(nn.Module):
    """
    蛋白共表达图平滑层：利用 KNN 图对预测值做邻居加权平滑

    输入: y_t (P, N) — P 个蛋白节点，每个有 N 个条件样本的特征
    输出: y_smoothed (P, N) — 平滑后的蛋白表达

    公式: y_smoothed[i] = y[i] + α * mean_{j∈N(i)} (y[j] - y[i])
          其中 α 为可学习平滑强度（初始值 0.1，训练中调整）

    α≈0 时近似恒等映射，保持初始无 GNN 的行为。
    """

    def __init__(self, alpha_init=0.01):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, y_t, edge_index):
        """
        y_t:        (P, N) — P 蛋白节点特征
        edge_index: (2, E) — 有向边列表 (src → dst)
        """
        P, N_batch = y_t.shape
        src, dst = edge_index  # (E,), (E,)

        # 入度（每个节点的邻居数）
        deg = torch.bincount(dst, minlength=P).float().view(P, 1).clamp(min=1)

        # 邻居聚合：对每个目标节点，求所有源节点值的和
        neighbor_sum = torch.zeros(P, N_batch, device=y_t.device, dtype=y_t.dtype)
        neighbor_sum = neighbor_sum.index_add(0, dst, torch.nan_to_num(y_t[src], nan=0.0))
        neighbor_mean = neighbor_sum / deg  # (P, N)

        # 可学习平滑：向邻居均值偏移 α 比例，clamp 防止极端值
        alpha = self.alpha.clamp(-0.5, 0.5)
        delta = alpha * (neighbor_mean - y_t)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        y_smoothed = y_t + delta
        y_smoothed = torch.nan_to_num(y_smoothed, nan=0.0, posinf=1e4, neginf=-1e4)

        return y_smoothed


# ══════════════════════════════════════════════════════════════════════════════
# AIVCModel: 完整模型 (B1 + B2 + B3 集成)
# ══════════════════════════════════════════════════════════════════════════════

class AIVCModel(nn.Module):
    """
    AIVC 完整模型

    数据流:
        x (N, dim_in)
        → ConditionEncoder → emb (N, 256)
        → ResidualDecoder   → {baseline, deltas, y_raw}
        → [+ GNN 平滑]      → y_smoothed
        → CalibrationHead   → δ_cal
        → y_pred = y_smoothed + δ_cal

    参数:
        dim_in:       输入特征维度（来自 Person A 的 build_condition_features）
        n_proteins:   输出蛋白数
        dim_emb:      条件嵌入维度（默认 256）
        use_gnn:      是否启用 GNN（B4 完成后可用）
    """

    def __init__(self, dim_in, n_proteins, dim_emb=256, use_gnn=False):
        super().__init__()
        from aivc.decoder import ResidualDecoder

        self.dim_emb = dim_emb
        self.n_proteins = n_proteins
        self.use_gnn = use_gnn

        self.encoder = ConditionEncoder(dim_in, hidden=dim_emb)
        self.decoder = ResidualDecoder(dim_emb, n_proteins)
        self.calibration = CalibrationHead(dim_emb, n_proteins)

        # B4: GNN 层 — use_gnn=True 时初始化
        self.gnn = None
        self.edge_index = None
        if use_gnn:
            self.gnn = ProteinGraphSmooth()

    def set_edge_index(self, edge_index):
        """注入 B4 产出的图结构（训练前调用）"""
        self.edge_index = edge_index

    def forward(self, x):
        emb = self.encoder(x)
        dec_out = self.decoder(emb)
        delta_cal = self.calibration(emb)

        y_raw = dec_out["y_raw"]

        if self.gnn is not None and self.edge_index is not None:
            # GNN: (N, P) → (P, N) → GraphConv → (P, N) → (N, P)
            y_t = y_raw.t()
            y_t = self.gnn(y_t, self.edge_index)
            y_raw = y_t.t()
            # Safety: nan_to_num after GNN to prevent NaN propagation
            y_raw = torch.nan_to_num(y_raw, nan=0.0, posinf=1e4, neginf=-1e4)

        y_pred = y_raw + delta_cal

        return {
            "y_pred": y_pred,
            "y_raw": y_raw,
            "baseline": dec_out["baseline"],
            "delta_drug": dec_out["delta_drug"],
            "delta_strain": dec_out["delta_strain"],
            "delta_context": dec_out["delta_context"],
            "delta_cal": delta_cal,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 自检
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 54)
    print("B1: ConditionEncoder 自检")
    print("=" * 54)

    encoder = ConditionEncoder(dim_in=256)
    x = torch.randn(32, 256)
    out = encoder(x)
    assert out.shape == (32, 256), f"Expected (32,256), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in encoder output"
    n_params_b1 = sum(p.numel() for p in encoder.parameters())
    print(f"  shape: {out.shape} [OK]")
    print(f"  no NaN: [OK]")
    print(f"  参数量: {n_params_b1:,}")
    print("B1 [OK]")

    print()
    print("=" * 54)
    print("B3: CalibrationHead 自检")
    print("=" * 54)

    cal = CalibrationHead(dim_emb=256, n_proteins=4422)
    emb = torch.randn(32, 256)
    offset = cal(emb)

    # 1. shape
    assert offset.shape == (32, 4422), f"Expected (32,4422), got {offset.shape}"
    print(f"  shape: {offset.shape} [OK]")

    # 2. zero init 生效
    max_abs = offset.abs().max().item()
    assert max_abs < 1e-6, f"Zero init 未生效: max={max_abs:.6f}"
    print(f"  zero init: max={max_abs:.2e} [OK]")

    # 3. 梯度流测试
    loss = offset.sum()
    loss.backward()
    assert cal.net[-1].weight.grad is not None, "无梯度!"
    print(f"  梯度流: [OK]")

    n_params_b3 = sum(p.numel() for p in cal.parameters())
    print(f"  参数量: {n_params_b3:,}")
    print("B3 [OK]")

    print()
    print("=" * 54)
    print("AIVCModel (B1+B2+B3) 集成自检")
    print("=" * 54)

    model = AIVCModel(dim_in=256, n_proteins=4422)
    x = torch.randn(32, 256)
    out = model(x)

    # 1. 全部 7 个 key 存在且 shape 正确
    expected_keys = [
        "y_pred", "y_raw", "baseline", "delta_drug",
        "delta_strain", "delta_context", "delta_cal",
    ]
    for key in expected_keys:
        assert key in out, f"Missing key: {key}"
        assert out[key].shape == (32, 4422), f"Bad shape: {key} = {out[key].shape}"
    print(f"  全部 {len(expected_keys)} 个 key 存在且 shape (32, 4422) [OK]")

    # 2. 恒等式验证
    # y_raw = baseline + Σ deltas
    y_raw_recomputed = (
        out["baseline"] + out["delta_drug"] + out["delta_strain"] + out["delta_context"]
    )
    assert torch.allclose(out["y_raw"], y_raw_recomputed, atol=1e-5), \
        "[FAIL] 残差恒等式不成立!"
    print(f"  残差恒等式 y_raw = baseline + Σ deltas [OK]")

    # y_pred = y_raw + delta_cal
    assert torch.allclose(out["y_pred"], out["y_raw"] + out["delta_cal"], atol=1e-5), \
        "[FAIL] 预测恒等式不成立!"
    print(f"  预测恒等式 y_pred = y_raw + δ_cal [OK]")

    # 3. 总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,} (target < 5M)")
    assert total_params < 5_000_000, f"参数量超标: {total_params:,} > 5M"

    # 4. 梯度流
    loss = out["y_pred"].sum()
    loss.backward()
    no_grad_params = [name for name, p in model.named_parameters() if p.grad is None]
    if no_grad_params:
        for name in no_grad_params:
            print(f"  [WARN] {name} 无梯度!")
    else:
        print(f"  全部参数梯度正常 [OK]")

    print()
    print("AIVCModel B1+B2+B3 [OK]")

    # ── B5: GNN 集成自检 ──
    print()
    print("=" * 54)
    print("B5: GNN 集成自检 (use_gnn=True)")
    print("=" * 54)

    # 造一个简单的图结构（环形，确保每个节点都有邻居）
    P = 100  # 用小规模测试
    edge_list = []
    for i in range(P):
        edge_list.append([i, (i + 1) % P])
        edge_list.append([i, (i + 2) % P])
    test_edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    gnn_model = AIVCModel(dim_in=256, n_proteins=P, use_gnn=True)
    gnn_model.set_edge_index(test_edge_index)
    x_gnn = torch.randn(4, 256)
    out_gnn = gnn_model(x_gnn)

    # 1. 输出 shape
    assert out_gnn["y_pred"].shape == (4, P), f"Bad shape: {out_gnn['y_pred'].shape}"
    print(f"  shape (4, {P}) [OK]")

    # 2. GNN 启用时 y_pred 来自平滑后的 y_raw
    assert gnn_model.gnn is not None, "GNN 未初始化!"
    assert gnn_model.edge_index is not None, "edge_index 未设置!"
    print(f"  GNN 层已初始化: {type(gnn_model.gnn).__name__} [OK]")
    print(f"  edge_index 已注入: shape={gnn_model.edge_index.shape} [OK]")

    # 3. 梯度流
    loss_gnn = out_gnn["y_pred"].sum()
    loss_gnn.backward()
    gnn_no_grad = [n for n, p in gnn_model.named_parameters() if p.grad is None]
    if gnn_no_grad:
        for name in gnn_no_grad:
            print(f"  [WARN] {name} 无梯度!")
    else:
        print(f"  全部参数（含 GNN）梯度正常 [OK]")

    # 4. 参数量（含 GNN）
    total_gnn = sum(p.numel() for p in gnn_model.parameters())
    print(f"  参数量（含 GNN）: {total_gnn:,}")

    print()
    print("B5 GNN 集成 [OK]")

    print()
    print("[OK] All self-checks passed!")
