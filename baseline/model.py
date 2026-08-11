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
