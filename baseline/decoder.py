"""
残差分解解码器模块 (B2)

ResidualDecoder: 条件嵌入 → 5头独立解码 → 残差求和
  - baseline_head:      共享基线响应
  - delta_drug_head:    化合物特异残差
  - delta_strain_head:  菌株调制残差
  - delta_context_head: 温度×时间×培养基上下文效应
  - 恒等式: y_raw = baseline + delta_drug + delta_strain + delta_context

SimpleDecoder: 端到端解码器（消融对照用）
  - 单头直接映射，参数量通过加宽 hidden 与 ResidualDecoder 拉齐
"""
import torch
import torch.nn as nn


class ResidualDecoder(nn.Module):
    """
    残差分解解码器：条件嵌入 → 5头独立解码 → 残差求和

    每个 head 用瓶颈结构 Linear(256→128) → Linear(128→n_proteins)，
    4 头总计约 3.0M 参数。

    输出 dict:
        baseline:      共享基线响应
        delta_drug:    化合物特异残差
        delta_strain:  菌株调制残差
        delta_context: 温度×时间×培养基上下文效应
        y_raw:         四项求和 (baseline + deltas)
    """

    def __init__(self, dim_emb=256, n_proteins=4422, hidden=128, dropout=0.1):
        super().__init__()
        self.dim_emb = dim_emb
        self.n_proteins = n_proteins

        def make_head():
            return nn.Sequential(
                nn.Linear(dim_emb, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_proteins),
            )

        self.baseline_head = make_head()
        self.delta_drug_head = make_head()
        self.delta_strain_head = make_head()
        self.delta_context_head = make_head()

    def forward(self, embedding):
        baseline = self.baseline_head(embedding)
        d_drug = self.delta_drug_head(embedding)
        d_strain = self.delta_strain_head(embedding)
        d_context = self.delta_context_head(embedding)

        y_raw = baseline + d_drug + d_strain + d_context

        return {
            "baseline": baseline,
            "delta_drug": d_drug,
            "delta_strain": d_strain,
            "delta_context": d_context,
            "y_raw": y_raw,
        }


class SimpleDecoder(nn.Module):
    """
    端到端解码器（消融对照用）—— 单头直接映射

    参数量通过加宽 hidden 与 ResidualDecoder 拉齐，
    确保消融对比公平。
    """

    def __init__(self, dim_emb=256, n_proteins=4422, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_emb, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_proteins),
        )

    def forward(self, embedding):
        y = self.net(embedding)
        return {
            "baseline": y,
            "delta_drug": torch.zeros_like(y),
            "delta_strain": torch.zeros_like(y),
            "delta_context": torch.zeros_like(y),
            "y_raw": y,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 自检
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 54)
    print("B2: ResidualDecoder + SimpleDecoder 自检")
    print("=" * 54)

    # --- 测试 ResidualDecoder ---
    print("\n[ResidualDecoder]")
    decoder = ResidualDecoder(dim_emb=256, n_proteins=4422)
    emb = torch.randn(32, 256)
    out = decoder(emb)

    # 1. shape
    for key in ["baseline", "delta_drug", "delta_strain", "delta_context", "y_raw"]:
        assert out[key].shape == (32, 4422), f"Bad shape: {key} = {out[key].shape}"
    print(f"  全部 5 个 key 的 shape = (32, 4422) [OK]")

    # 2. 恒等式（核心约束！）
    y_raw_recomputed = (
        out["baseline"] + out["delta_drug"] + out["delta_strain"] + out["delta_context"]
    )
    assert torch.allclose(out["y_raw"], y_raw_recomputed, atol=1e-5), \
        "[FAIL] 残差恒等式不成立!"
    print(f"  残差恒等式 y_raw = baseline + Σ deltas [OK]")

    # 3. 参数量
    n_residual = sum(p.numel() for p in decoder.parameters())
    print(f"  参数量: {n_residual:,} (预期 ~3.0M)")

    # --- 测试 SimpleDecoder ---
    print("\n[SimpleDecoder]")
    simp = SimpleDecoder(dim_emb=256, n_proteins=4422)
    out2 = simp(emb)
    n_simple = sum(p.numel() for p in simp.parameters())
    print(f"  参数量: {n_simple:,}")

    # SimpleDecoder 输出也兼容 dict 接口
    for key in ["baseline", "delta_drug", "delta_strain", "delta_context", "y_raw"]:
        assert key in out2, f"SimpleDecoder missing key: {key}"
        assert out2[key].shape == (32, 4422), f"Bad shape: {key} = {out2[key].shape}"
    assert torch.allclose(out2["y_raw"], out2["baseline"], atol=1e-5), \
        "SimpleDecoder: y_raw should equal baseline"
    assert out2["delta_drug"].abs().max() < 1e-6, "SimpleDecoder: delta_drug should be 0"
    print(f"  dict 接口兼容 [OK]")
    print(f"  delta heads = 0 [OK]")

    # --- 参数量对比 ---
    print(f"\n  参数量差异: {abs(n_residual - n_simple) / max(n_residual, n_simple) * 100:.1f}%")

    print("\nB2 [OK]")
    print("\n[OK] All self-checks passed!")
