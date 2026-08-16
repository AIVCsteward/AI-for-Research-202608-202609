"""
模型定义模块：条件编码 MLP

当前实现（Phase 1 baseline）:
  ConditionMLP: 2 隐藏层 MLP (256→256)，one-hot 条件特征 → 全蛋白组预测

升级方向（设计文档 Phase 2+）:
  - Encoder-Decoder 架构（编码器→条件嵌入→解码器）
  - 残差分解解码器：Baseline Head + Δ_drug Head + Δ_strain Head + Δ_context Head
  - 蛋白共表达 GNN 平滑层
  - 批次校准分支 (Calibration Head)
  - 每个 Δ 头可复用当前 ConditionMLP 的 2 层结构作为子模块
"""
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
