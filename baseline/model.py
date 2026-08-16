"""
基线模型：简单的条件编码 MLP（Phase 1 baseline）

ConditionMLP: one-hot 特征 → 2 隐层 → N_proteins 输出

注意：正式模型（AIVCModel, ConditionEncoder, ResidualDecoder, GNN 等）
已迁移至 aivc/ 目录。本模块仅保留最简基线模型供对标使用。
"""
import torch
import torch.nn as nn


class ConditionMLP(nn.Module):
    """
    条件编码 MLP: one-hot 特征 → 2 隐层 → N_proteins 输出

    架构:
        Linear(dim_in, 256) → ReLU → BatchNorm → Dropout(0.1)
      → Linear(256, 256)    → ReLU → BatchNorm → Dropout(0.1)
      → Linear(256, dim_out)

    参数:
        dim_in:   输入特征维度（取决于特征工程）
        dim_out:  输出蛋白数
        hidden:   隐藏层宽度（默认 256）
        dropout:  Dropout 比例（默认 0.1）
    """

    def __init__(self, dim_in, dim_out, hidden=256, dropout=0.1,
                 protein_prior=None, use_bias=False):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, dim_out)
        self.use_bias = use_bias
        if use_bias:
            self.b_p = nn.Parameter(torch.zeros(dim_out))
        if protein_prior is not None:
            # (dim_out, d) 蛋白先验矩阵，冻结；W_a 零初始化 → 第 0 步等价无先验
            self.register_buffer(
                "E", torch.as_tensor(protein_prior, dtype=torch.float32)
            )
            self.W_a = nn.Linear(hidden, self.E.shape[1])
            nn.init.zeros_(self.W_a.weight)
            nn.init.zeros_(self.W_a.bias)

    def forward(self, x):
        h = self.trunk(x)
        y = self.head(h)
        if hasattr(self, "E"):
            y = y + self.W_a(h) @ self.E.T
        if self.use_bias:
            y = y + self.b_p
        return y


class SplitConditionMLP(nn.Module):
    """拆分方案 A：输入端按语义拆成「上下文」和「化药+菌株」两个子网络，输出相加。

    forward 接收拼接输入 x = [x_ctx | x_bio]，内部拆成两半分别过子网络再相加：
        y = f_ctx(x[:, :ctx_dim]) + f_bio(x[:, ctx_dim:])
    """

    def __init__(self, ctx_dim, bio_dim, dim_out, hidden=256, dropout=0.1):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.ctx_net = ConditionMLP(ctx_dim, dim_out, hidden, dropout)
        self.bio_net = ConditionMLP(bio_dim, dim_out, hidden, dropout)

    def forward(self, x):
        x_ctx = x[:, : self.ctx_dim]
        x_bio = x[:, self.ctx_dim :]
        return self.ctx_net(x_ctx) + self.bio_net(x_bio)


class ControlTreatSplitModel(nn.Module):
    """拆分方案 B：对照模型学基线 + 扰动模型学 Δ。

    y_control  = ctrl_net(x)
    y_treatment = ctrl_net(x) + treat_net(x)
    """

    def __init__(self, dim_in, dim_out, hidden=256, dropout=0.1):
        super().__init__()
        self.ctrl_net = ConditionMLP(dim_in, dim_out, hidden, dropout)
        self.treat_net = ConditionMLP(dim_in, dim_out, hidden, dropout)

    def forward_ctrl(self, x):
        return self.ctrl_net(x)

    def forward_treat(self, x):
        return self.ctrl_net(x) + self.treat_net(x)


# ══════════════════════════════════════════════════════════════════════════════
# 自检
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 54)
    print("ConditionMLP 自检")
    print("=" * 54)

    model = ConditionMLP(dim_in=128, dim_out=4422)
    x = torch.randn(32, 128)
    out = model(x)
    assert out.shape == (32, 4422), f"Expected (32,4422), got {out.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  shape: {out.shape} [OK]")
    print(f"  参数量: {n_params:,}")
    print("ConditionMLP [OK]")
