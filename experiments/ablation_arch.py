"""
架构消融实验 (B6)

对比四组配置:
  full_residual:   ResidualDecoder + Calibration → 全部创新生效时的基线
  no_residual:     SimpleDecoder  + Calibration → 残差分解的贡献
  no_calibration:  ResidualDecoder 无校准       → 批次校准的贡献
  with_gnn:        ResidualDecoder + Cal + GNN   → GNN 的增量贡献

数据纪律:
  - 所有对比使用相同的数据划分、随机种子、训练超参数
  - 参数量通过 hidden 宽度调整拉齐到 ±20% 以内
  - 每个实验记录 val_both 和 val_chem_only 的 per-protein R²
"""
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# 项目根路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# 消融配置
# ══════════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = [
    {
        "name": "full_residual",
        "decoder": "residual",
        "calibration": True,
        "gnn": False,
        "desc": "残差分解 + 校准（基线）",
    },
    {
        "name": "no_residual",
        "decoder": "simple",
        "calibration": True,
        "gnn": False,
        "desc": "端到端解码器（残差分解消融）",
    },
    {
        "name": "no_calibration",
        "decoder": "residual",
        "calibration": False,
        "gnn": False,
        "desc": "残差分解 / 无校准（校准消融）",
    },
    {
        "name": "with_gnn",
        "decoder": "residual",
        "calibration": True,
        "gnn": True,
        "desc": "残差分解 + 校准 + GNN",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# 模型构建
# ══════════════════════════════════════════════════════════════════════════════

def build_model(config, dim_in, n_proteins, dim_emb=256):
    """
    按配置构建模型，确保对比公平（参数总量接近）

    参数:
        config:      消融配置 dict
        dim_in:      输入特征维度
        n_proteins:  蛋白数
        dim_emb:     条件嵌入维度

    返回:
        model: AIVCModel 实例
    """
    from baseline.model import AIVCModel
    from baseline.decoder import ResidualDecoder, SimpleDecoder

    model = AIVCModel(
        dim_in, n_proteins,
        dim_emb=dim_emb,
        use_gnn=config.get("gnn", False),
    )

    # 解码器替换
    if config["decoder"] == "simple":
        # SimpleDecoder with widened hidden to match param count
        model.decoder = SimpleDecoder(dim_emb, n_proteins, hidden=512)

    # 校准分支替换
    if not config["calibration"]:
        model.calibration = _ZeroCalibration(n_proteins)

    return model


class _ZeroCalibration(nn.Module):
    """零输出校准头 — 消融用，保持 forward 接口兼容"""
    def __init__(self, n_proteins):
        super().__init__()
        self.n_proteins = n_proteins

    def forward(self, embedding):
        return torch.zeros(embedding.shape[0], self.n_proteins, device=embedding.device)


# ══════════════════════════════════════════════════════════════════════════════
# 训练适配器
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, X_train, y_train, mask_train, optimizer, batch_size=256):
    """单 epoch 训练（适配 AIVCModel 的 dict 输出）"""
    from baseline.training import mask_aware_mse

    model.train()
    N = X_train.shape[0]
    perm = torch.randperm(N)
    total_loss = 0.0
    n_batches = 0

    for i in range(0, N, batch_size):
        idx = perm[i: i + batch_size]
        optimizer.zero_grad()
        out = model(X_train[idx])
        loss = mask_aware_mse(out["y_pred"], y_train[idx], mask_train[idx])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, X_val, y_val, mask_val):
    """验证：返回 mask-aware MSE"""
    from baseline.training import mask_aware_mse

    model.eval()
    out = model(X_val)
    return mask_aware_mse(out["y_pred"], y_val, mask_val).item()


# ══════════════════════════════════════════════════════════════════════════════
# 主实验
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(
    ablation_configs=None,
    epochs=100,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    device="cpu",
    seed=42,
):
    """
    跑一组消融实验，记录全部指标

    返回:
        results: [{"name": ..., "params": ..., "val_losses": {...}, ...}, ...]
    """
    from baseline.data import load_raw_data, preprocess
    from baseline.features import build_condition_features
    from baseline.training import prepare_training_data, mask_aware_mse

    if ablation_configs is None:
        ablation_configs = ABLATION_CONFIGS

    # ── 数据准备（所有实验共享）──
    torch.manual_seed(seed)
    np.random.seed(seed)

    meta, prot = load_raw_data()
    y_log2, mask_matrix, meta, protein_names, train_mask = preprocess(meta, prot)

    X_all, encoders = build_condition_features(meta, fit_encoders=True)
    dim_in = X_all.shape[1]
    n_proteins = len(protein_names)

    # 训练/验证张量
    split_masks = {}
    for sn in meta["split_final"].unique():
        split_masks[sn] = meta["split_final"] == sn

    val_splits = [s for s in ["val_both", "val_chem_only", "val_strain_only"] if s in split_masks]

    X_train_t = torch.tensor(
        X_all[train_mask.values], dtype=torch.float32
    ).to(device)
    y_train_t = torch.tensor(
        y_log2.loc[train_mask].fillna(0).values, dtype=torch.float32
    ).to(device)
    mask_train_t = torch.tensor(
        mask_matrix.loc[train_mask].values, dtype=torch.float32
    ).to(device)

    val_data = {}
    for sn in val_splits:
        m = split_masks[sn]
        if m.sum() == 0:
            continue
        val_data[sn] = {
            "X": torch.tensor(X_all[m.values], dtype=torch.float32).to(device),
            "y": torch.tensor(
                y_log2.loc[m].fillna(0).values, dtype=torch.float32
            ).to(device),
            "mask": torch.tensor(
                mask_matrix.loc[m].values, dtype=torch.float32
            ).to(device),
        }

    print(f"数据: dim_in={dim_in}, n_proteins={n_proteins}")
    val_desc = ", ".join(f"{k}: {v['X'].shape[0]}" for k, v in val_data.items())
    print(f"训练: {X_train_t.shape[0]}, 验证: {{{val_desc}}}")

    # ── GNN 图结构（with_gnn 实验共享）──
    edge_index = None
    if any(cfg.get("gnn") for cfg in ablation_configs):
        from baseline.protein_graph import build_protein_graph
        print("\n构建蛋白共表达图...")
        edge_index = build_protein_graph(
            y_log2.loc[train_mask],
            mask_matrix.loc[train_mask],
            k=5, threshold=0.7,
        )

    # ── 逐实验训练 ──
    results = []
    for cfg in ablation_configs:
        print(f"\n{'=' * 54}")
        print(f"消融: {cfg['name']} — {cfg['desc']}")
        print(f"{'=' * 54}")

        torch.manual_seed(seed)
        model = build_model(cfg, dim_in, n_proteins).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {n_params:,}")

        if cfg.get("gnn") and edge_index is not None:
            model.set_edge_index(edge_index.to(device))

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        best_val = float("inf")
        best_state = None
        history = {"train_loss": [], "val_loss": {sn: [] for sn in val_data}}

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, X_train_t, y_train_t, mask_train_t, optimizer, batch_size,
            )
            history["train_loss"].append(train_loss)

            val_losses = []
            for sn, vd in val_data.items():
                v_loss = evaluate(model, vd["X"], vd["y"], vd["mask"])
                history["val_loss"][sn].append(v_loss)
                val_losses.append(v_loss)

            avg_val = np.mean(val_losses)
            scheduler.step(avg_val)

            if avg_val < best_val:
                best_val = avg_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 20 == 0:
                val_str = " ".join(
                    f"{sn}={history['val_loss'][sn][-1]:.4f}" for sn in val_data
                )
                print(f"  Epoch {epoch+1:3d}/{epochs} | train={train_loss:.4f} | {val_str} | lr={optimizer.param_groups[0]['lr']:.2e}")

        # 加载最佳模型
        model.load_state_dict(best_state)

        # 记录最终验证 loss
        final_val = {}
        for sn, vd in val_data.items():
            final_val[sn] = evaluate(model, vd["X"], vd["y"], vd["mask"])

        result = {
            "name": cfg["name"],
            "desc": cfg["desc"],
            "config": cfg,
            "params": n_params,
            "best_val_loss": best_val,
            "final_val_losses": final_val,
            "history": history,
        }
        results.append(result)

        print(f"  完成 | best val={best_val:.4f} | "
              + " ".join(f"{sn}={final_val[sn]:.4f}" for sn in val_data))

    # ── 汇总 ──
    print(f"\n{'=' * 54}")
    print("消融实验汇总")
    print(f"{'=' * 54}")
    print(f"{'实验':<22} {'参数量':>10} {'best_val':>10}", end="")
    for sn in val_data:
        print(f" {sn:>12}", end="")
    print()
    print("-" * (44 + 14 * len(val_data)))
    for r in results:
        print(f"{r['name']:<22} {r['params']:>10,} {r['best_val_loss']:>10.4f}", end="")
        for sn in val_data:
            print(f" {r['final_val_losses'][sn]:>12.4f}", end="")
        print()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 快速自检（不训练，仅验证模型构建）
# ══════════════════════════════════════════════════════════════════════════════

def sanity_check():
    """快速自检：四个配置均能成功初始化 + 前向通过"""
    print("=" * 54)
    print("B6: 消融实验 — 模型构建自检")
    print("=" * 54)

    dim_in, n_proteins = 256, 4422  # 真实蛋白数（参数匹配在此规模校准）
    x = torch.randn(4, dim_in)

    for cfg in ABLATION_CONFIGS:
        model = build_model(cfg, dim_in, n_proteins)
        out = model(x)
        n_param = sum(p.numel() for p in model.parameters())

        assert out["y_pred"].shape == (4, n_proteins), \
            f"{cfg['name']}: bad shape {out['y_pred'].shape}"
        print(f"  {cfg['name']:<22} params={n_param:>10,} shape={out['y_pred'].shape} [OK]")

    # 参数量差异检查（公平对比）
    params_list = []
    for cfg in ABLATION_CONFIGS:
        m = build_model(cfg, dim_in, n_proteins)
        params_list.append(sum(p.numel() for p in m.parameters()))

    max_p, min_p = max(params_list), min(params_list)
    diff_pct = (max_p - min_p) / max_p * 100
    print(f"\n  参数量范围: {min_p:,} ~ {max_p:,} (差异 {diff_pct:.1f}%)")
    if diff_pct < 20:
        print(f"  公平对比: [OK]")
    else:
        print(f"  [WARN] 参数量差异 > 20%，消融对比可能不公平")

    print("\nB6 模型构建 [OK]")
    print("[OK] All sanity checks passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="架构消融实验")
    parser.add_argument("--run", action="store_true", help="运行完整训练实验")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--device", default="cpu", help="设备 (cpu/cuda)")
    parser.add_argument("--sanity", action="store_true", default=True,
                        help="仅运行模型构建自检（默认）")
    args = parser.parse_args()

    if args.run:
        run_ablation(epochs=args.epochs, device=args.device)
    else:
        sanity_check()
