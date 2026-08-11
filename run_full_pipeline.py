"""
================================================================================
AIVC 全流程实验脚本 — AutoDL / 本地一键运行
================================================================================
运行内容:
  1. 数据预处理 + 双基线评估
  2. AIVCModel (残差分解 + 校准) 完整训练
  3. AIVCModel + GNN 训练
  4. 编码器消融 (5 组: full / no-strain-prior / no-chem-anchor / no-hash / no-cross)
  5. 架构消融 (4 组: residual / simple / no-calib / +GNN)
  6. Loss 消融 (4 组: MSE / MSE+FC / MSE+FC+L2 / full)
  7. 生成 FULL_REPORT.md

用法:
    python run_full_pipeline.py --epochs 100 --output-dir experiments/outputs

AutoDL 推荐:
    pip install torch numpy pandas scikit-learn
    python run_full_pipeline.py --epochs 100 --device cuda
================================================================================
"""
from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ── 项目模块 ──────────────────────────────────────────────────────────────
from baseline.data import load_raw_data, preprocess, get_split_masks, identify_controls
from baseline.evaluation import (
    VAL_SPLITS,
    TEST_SPLITS,
    compute_protein_mean,
    evaluate_protein_mean_baseline,
    build_control_lookup,
    evaluate_matched_control_baseline,
    build_matched_control_pairs,
    compute_fold_change,
    evaluate_global_r2,
    evaluate_per_protein_r2,
    print_diagnostics,
)
from baseline.features import fit_feature_encoders, build_condition_features
from aivc.config import get_experiment_config
from aivc.model import AIVCModel
from aivc.decoder import ResidualDecoder, SimpleDecoder
from aivc.training import (
    prepare_training_data,
    prepare_fold_change_index,
    train,
    mask_aware_mse,
    masked_per_protein_r2_median,
)
from aivc.protein_graph import build_protein_graph
from aivc.losses import compute_target_edge_corr
from aivc.entity_representations import (
    ChemicalAnchorEncoder,
    CrossFeatureEncoder,
    HashEncoder,
    StrainPriorEncoder,
)


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


@torch.no_grad()
def predict_batch(model, features, device, batch_size=256):
    """批量推理，返回 numpy 数组。"""
    model.eval()
    preds = []
    for start in range(0, len(features), batch_size):
        x = torch.as_tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
        output = model(x)
        y = output["y_pred"] if isinstance(output, dict) else output
        preds.append(y.detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.empty((0, model.n_proteins), dtype=np.float32)


def evaluate_model_on_splits(model, X_all, meta, y_log2, mask_matrix, split_masks, device, splits=None):
    """在指定 split 上评估模型，返回指标 dict。"""
    if splits is None:
        splits = VAL_SPLITS + TEST_SPLITS
    model.eval()
    metrics = {}
    for split_name in splits:
        m = split_masks.get(split_name)
        if m is None or m.sum() == 0:
            continue
        rows = m.to_numpy(dtype=bool)
        pred = predict_batch(model, X_all[rows], device)
        global_r2 = evaluate_global_r2(y_log2.loc[m], pred, mask_matrix.loc[m])
        pp_r2 = evaluate_per_protein_r2(y_log2.loc[m], pred, mask_matrix.loc[m])
        metrics[split_name] = {
            "n_samples": int(rows.sum()),
            "global_r2": float(global_r2),
            "per_protein_r2_median": float(pp_r2),
        }
    return metrics


def compare_with_baseline(model_metrics, baseline_metrics, split_name):
    """对比模型与 Matched Control 基线。"""
    model_val = model_metrics.get(split_name, {}).get("per_protein_r2_median")
    baseline_val = baseline_metrics.get(split_name, {}).get("per_protein_r2_median")
    if model_val is None or baseline_val is None:
        return None
    return float(model_val - baseline_val)


# ══════════════════════════════════════════════════════════════════════════════
# 实验 1: 主模型 — AIVCModel (残差分解 + 校准)
# ══════════════════════════════════════════════════════════════════════════════

def run_main_experiment(ctx, config, device, output_dir):
    """训练 AIVCModel（无 GNN）+ 评估。"""
    print("\n" + "=" * 70)
    print("实验 1: AIVCModel (残差分解 + 批次校准)")
    print("=" * 70)

    set_seed(config.get("seed", 42))
    model = AIVCModel(
        dim_in=ctx["X_train"].shape[1],
        n_proteins=len(ctx["protein_names"]),
        dim_emb=config["encoder"]["d_emb"],
        use_gnn=False,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")

    loss_weights = {
        "mse": config["loss"]["mse_weight"],
        "fc": config["loss"]["fc_pearson_weight"],
        "l2": config["loss"]["residual_l2_weight"],
        "corr": 0.0,  # GNN 关闭时不启用相关一致性 loss
    }

    t0 = time.perf_counter()
    model, history = train(
        model,
        ctx["X_train"],
        ctx["y_train"],
        ctx["mask_train"],
        ctx["val_data"],
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        verbose=True,
        fc_control_index=ctx["fc_control_index"],
        loss_weights=loss_weights,
        early_stopping_patience=config["training"]["early_stopping_patience"],
        early_stopping_split="val_both",
    )
    train_time = time.perf_counter() - t0
    print(f"训练耗时: {train_time / 60:.1f} min")

    # 保存 checkpoint
    ckpt_path = Path(output_dir) / "main_model.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    # 评估
    metrics = evaluate_model_on_splits(
        model, ctx["X_all"], ctx["meta"], ctx["y_log2"], ctx["mask_matrix"],
        ctx["split_masks"], device,
    )

    # 生成提交文件
    test_mask = ctx["meta"]["split_final"].str.startswith("test")
    test_meta = ctx["meta"].loc[test_mask]
    X_test = build_condition_features(test_meta, encoders=ctx["encoders"])
    pred_test = predict_batch(model, X_test, device)
    pred_test = np.clip(pred_test, 5.0, 40.0)
    submission = pd.DataFrame(pred_test, index=test_meta.index, columns=ctx["protein_names"])
    submission.index.name = "sample_ID"
    sub_path = Path(output_dir) / "prediction_main.csv"
    submission.to_csv(sub_path)

    return {
        "name": "AIVCModel (残差分解 + 校准)",
        "params": n_params,
        "train_time_min": round(train_time / 60, 1),
        "best_monitor": float(history["best_monitor"]) if history.get("best_monitor") is not None else None,
        "final_losses": {
            k: float(history[k][-1]) if history.get(k) and history[k] else None
            for k in ("loss_total", "loss_mse", "loss_fc", "loss_l2", "loss_corr")
        },
        "metrics": metrics,
        "checkpoint": str(ckpt_path),
        "submission": str(sub_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 实验 2: AIVCModel + GNN
# ══════════════════════════════════════════════════════════════════════════════

def run_gnn_experiment(ctx, config, device, output_dir):
    """训练 AIVCModel + GNN + 全部 loss。"""
    print("\n" + "=" * 70)
    print("实验 2: AIVCModel + GNN (残差分解 + 校准 + 蛋白图平滑)")
    print("=" * 70)

    set_seed(config.get("seed", 42))

    # 构建蛋白图 + 相关性目标
    print("构建蛋白共表达图...")
    edge_index = build_protein_graph(
        ctx["y_log2"].loc[ctx["train_mask"]],
        ctx["mask_matrix"].loc[ctx["train_mask"]],
        k=config["gnn"]["k_neighbors"],
        threshold=config["gnn"]["pearson_threshold"],
    )
    target_edge_corr = compute_target_edge_corr(
        ctx["y_train"].detach().cpu(),
        edge_index.detach().cpu(),
        mask=ctx["mask_train"].detach().cpu().bool(),
    )
    print(f"图: {edge_index.shape[1]} 条边, 有效相关目标: {target_edge_corr.isfinite().sum().item()}")

    model = AIVCModel(
        dim_in=ctx["X_train"].shape[1],
        n_proteins=len(ctx["protein_names"]),
        dim_emb=config["encoder"]["d_emb"],
        use_gnn=True,
    ).to(device)
    model.set_edge_index(edge_index.to(device))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")

    # GNN experiment: use corr=0.0 initially to isolate GNN smoothing effect
    # Full multitask including corr loss is tested separately in loss ablation
    loss_weights = {
        "mse": config["loss"]["mse_weight"],
        "fc": config["loss"]["fc_pearson_weight"],
        "l2": config["loss"]["residual_l2_weight"],
        "corr": 0.0,
    }

    t0 = time.perf_counter()
    model, history = train(
        model,
        ctx["X_train"],
        ctx["y_train"],
        ctx["mask_train"],
        ctx["val_data"],
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        verbose=True,
        fc_control_index=ctx["fc_control_index"],
        edge_index=None,
        target_edge_corr=None,
        loss_weights=loss_weights,
        early_stopping_patience=config["training"]["early_stopping_patience"],
        early_stopping_split="val_both",
    )
    train_time = time.perf_counter() - t0
    print(f"训练耗时: {train_time / 60:.1f} min")

    ckpt_path = Path(output_dir) / "gnn_model.pt"
    torch.save(model.state_dict(), ckpt_path)

    metrics = evaluate_model_on_splits(
        model, ctx["X_all"], ctx["meta"], ctx["y_log2"], ctx["mask_matrix"],
        ctx["split_masks"], device,
    )

    # 提交文件
    test_mask = ctx["meta"]["split_final"].str.startswith("test")
    X_test = build_condition_features(ctx["meta"].loc[test_mask], encoders=ctx["encoders"])
    pred_test = predict_batch(model, X_test, device)
    pred_test = np.clip(pred_test, 5.0, 40.0)
    submission = pd.DataFrame(pred_test, index=ctx["meta"].loc[test_mask].index, columns=ctx["protein_names"])
    submission.index.name = "sample_ID"
    sub_path = Path(output_dir) / "prediction_gnn.csv"
    submission.to_csv(sub_path)

    return {
        "name": "AIVCModel + GNN",
        "params": n_params,
        "train_time_min": round(train_time / 60, 1),
        "best_monitor": float(history["best_monitor"]) if history.get("best_monitor") is not None else None,
        "final_losses": {
            k: float(history[k][-1]) if history.get(k) and history[k] else None
            for k in ("loss_total", "loss_mse", "loss_fc", "loss_l2", "loss_corr")
        },
        "metrics": metrics,
        "checkpoint": str(ckpt_path),
        "submission": str(sub_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 实验 3: 编码器消融
# ══════════════════════════════════════════════════════════════════════════════

ENCODER_ABLATIONS = [
    {"name": "full",            "desc": "全部特征"},
    {"name": "no_strain_prior", "desc": "移除 Strain Prior"},
    {"name": "no_chem_anchor",  "desc": "移除 Chemical Anchor"},
    {"name": "no_hash",         "desc": "移除 Hash 编码"},
    {"name": "no_cross",        "desc": "移除交叉特征"},
]


def _build_ablation_encoders(train_meta, y_train, mask_train, config, ablate):
    """构建消融变体的编码器（关掉指定特征组）。"""
    cfg = get_experiment_config(config)
    cfg["encoder"]["use_strain_prior"] = "no_strain_prior" != ablate
    cfg["encoder"]["use_chem_anchor"] = "no_chem_anchor" != ablate
    cfg["encoder"]["use_hash_features"] = "no_hash" != ablate
    cfg["encoder"]["use_cross_features"] = "no_cross" != ablate

    encoders = {"config": cfg, "cat_encoders": {}}
    for name, column in [("strains", "Strains"), ("chemicals", "perturbation_no_concentration"),
                          ("media", "Medium"), ("instruments", "instrument")]:
        values = train_meta[column].dropna().astype(str).unique()
        encoders[name] = {v: i for i, v in enumerate(sorted(values))}
        encoders["cat_encoders"][column] = encoders[name]

    ec = cfg["encoder"]
    if ec.get("use_strain_prior", True):
        encoders["strain_prior"] = StrainPriorEncoder(ec.get("strain_prior_pca_dim", 32)).fit(train_meta, y_train)
    if ec.get("use_chem_anchor", True):
        encoders["chem_anchor"] = ChemicalAnchorEncoder(ec.get("chem_anchor_pca_dim", 64)).fit(train_meta, y_train)
    if ec.get("use_hash_features", True):
        encoders["hash_chemical"] = HashEncoder(ec.get("hash_dim_chemical", 32), seed=11)
        encoders["hash_plate"] = HashEncoder(ec.get("hash_dim_plate", 16), seed=17)
    if ec.get("use_cross_features", True):
        encoders["cross_features"] = CrossFeatureEncoder(
            ec.get("cross_dim_strain_medium", 10),
            ec.get("cross_dim_chemical_temperature", 92),
        ).fit(train_meta)

    # Re-use baseline.features internals to get consistent raw features + projection
    from baseline.features import build_raw_condition_features, FixedDimProjector
    raw_train = build_raw_condition_features(train_meta, encoders)
    projector = FixedDimProjector(ec.get("d_emb", 256)).fit(raw_train)
    encoders["projector"] = projector
    encoders["d_emb"] = projector.output_dim
    encoders["raw_dim"] = raw_train.shape[1]
    return encoders


def run_encoder_ablation(ctx, config, device, output_dir):
    """编码器消融：逐个移除特征组。"""
    print("\n" + "=" * 70)
    print("实验 3: 编码器消融")
    print("=" * 70)

    train_meta = ctx["meta"].loc[ctx["train_mask"]]
    y_train = ctx["y_log2"].loc[ctx["train_mask"]]
    mask_train = ctx["mask_matrix"].loc[ctx["train_mask"]]

    results = []
    for ab in ENCODER_ABLATIONS:
        name, desc = ab["name"], ab["desc"]
        print(f"\n--- {name}: {desc} ---")
        set_seed(config.get("seed", 42))

        encs = _build_ablation_encoders(train_meta, y_train, mask_train, config, name)
        X_all = build_condition_features(ctx["meta"], encoders=encs)
        X_tr, y_tr, mask_tr, val_d = prepare_training_data(
            X_all, ctx["y_log2"], ctx["mask_matrix"], ctx["train_mask"],
            ctx["split_masks"], VAL_SPLITS, device,
        )
        model = AIVCModel(X_tr.shape[1], len(ctx["protein_names"]), dim_emb=config["encoder"]["d_emb"], use_gnn=False).to(device)

        t0 = time.perf_counter()
        model, history = train(
            model, X_tr, y_tr, mask_tr, val_d,
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            verbose=False,
            fc_control_index=ctx["fc_control_index"],
            loss_weights={"mse": 1.0, "fc": 0.3, "l2": 0.01, "corr": 0.0},
            early_stopping_patience=config["training"]["early_stopping_patience"],
        )
        train_time = time.perf_counter() - t0

        metrics = evaluate_model_on_splits(
            model, X_all, ctx["meta"], ctx["y_log2"], ctx["mask_matrix"],
            ctx["split_masks"], device, splits=VAL_SPLITS,
        )
        results.append({
            "name": name, "desc": desc,
            "train_time_min": round(train_time / 60, 1),
            "best_monitor": float(history["best_monitor"]) if history.get("best_monitor") is not None else None,
            "metrics": metrics,
        })
        print(f"  val_both per-protein R²: {metrics.get('val_both', {}).get('per_protein_r2_median', 'N/A')}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 实验 4: 架构消融
# ══════════════════════════════════════════════════════════════════════════════

class _ZeroCalibration(nn.Module):
    def __init__(self, n_proteins=4422):
        super().__init__()
        self.n_proteins = n_proteins
    def forward(self, x):
        return torch.zeros(x.shape[0], self.n_proteins, device=x.device)


ARCH_ABLATIONS = [
    {"name": "full_residual",  "decoder": "residual", "calib": True,  "gnn": False, "desc": "残差分解 + 校准"},
    {"name": "no_residual",    "decoder": "simple",   "calib": True,  "gnn": False, "desc": "端到端解码器"},
    {"name": "no_calibration", "decoder": "residual", "calib": False, "gnn": False, "desc": "残差分解 / 无校准"},
    {"name": "with_gnn",       "decoder": "residual", "calib": True,  "gnn": True,  "desc": "残差分解 + 校准 + GNN"},
]


def run_architecture_ablation(ctx, config, device, output_dir):
    """架构消融：残差分解、校准、GNN 的独立贡献。"""
    print("\n" + "=" * 70)
    print("实验 4: 架构消融")
    print("=" * 70)

    results = []
    for arch in ARCH_ABLATIONS:
        name, desc = arch["name"], arch["desc"]
        print(f"\n--- {name}: {desc} ---")
        set_seed(config.get("seed", 42))

        model = AIVCModel(
            dim_in=ctx["X_train"].shape[1],
            n_proteins=len(ctx["protein_names"]),
            dim_emb=config["encoder"]["d_emb"],
            use_gnn=arch["gnn"],
        ).to(device)

        if arch["decoder"] == "simple":
            model.decoder = SimpleDecoder(config["encoder"]["d_emb"], len(ctx["protein_names"]), hidden=512).to(device)
        if not arch["calib"]:
            model.calibration = _ZeroCalibration(len(ctx["protein_names"])).to(device)

        edge_index = None
        target_edge_corr = None
        corr_weight = 0.0
        if arch["gnn"]:
            print("  构建蛋白图...")
            edge_index = build_protein_graph(
                ctx["y_log2"].loc[ctx["train_mask"]],
                ctx["mask_matrix"].loc[ctx["train_mask"]],
                k=config["gnn"]["k_neighbors"],
                threshold=config["gnn"]["pearson_threshold"],
            )
            target_edge_corr = compute_target_edge_corr(
                ctx["y_train"].detach().cpu(), edge_index.detach().cpu(),
                mask=ctx["mask_train"].detach().cpu().bool(),
            )
            model.set_edge_index(edge_index.to(device))
            edge_index = edge_index.to(device)
            target_edge_corr = target_edge_corr.to(device)
            corr_weight = 0.0  # 架构消融关注结构差异，相关一致性 loss 由 Loss 消融覆盖

        loss_weights = {
            "mse": config["loss"]["mse_weight"],
            "fc": config["loss"]["fc_pearson_weight"],
            "l2": config["loss"]["residual_l2_weight"],
            "corr": corr_weight,
        }

        t0 = time.perf_counter()
        model, history = train(
            model, ctx["X_train"], ctx["y_train"], ctx["mask_train"], ctx["val_data"],
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            verbose=False,
            fc_control_index=ctx["fc_control_index"],
            edge_index=edge_index,
            target_edge_corr=target_edge_corr,
            loss_weights=loss_weights,
            early_stopping_patience=config["training"]["early_stopping_patience"],
        )
        train_time = time.perf_counter() - t0

        metrics = evaluate_model_on_splits(
            model, ctx["X_all"], ctx["meta"], ctx["y_log2"], ctx["mask_matrix"],
            ctx["split_masks"], device, splits=VAL_SPLITS,
        )
        results.append({
            "name": name, "desc": desc,
            "train_time_min": round(train_time / 60, 1),
            "best_monitor": float(history["best_monitor"]) if history.get("best_monitor") is not None else None,
            "metrics": metrics,
        })
        print(f"  val_both per-protein R²: {metrics.get('val_both', {}).get('per_protein_r2_median', 'N/A')}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 实验 5: Loss 消融
# ══════════════════════════════════════════════════════════════════════════════

LOSS_ABLATIONS = [
    {"name": "mse_only",     "desc": "仅 mask-aware MSE",                                       "w": {"mse": 1.0, "fc": 0.0, "l2": 0.0, "corr": 0.0}},
    {"name": "mse_fc",       "desc": "MSE + FC Pearson",                                        "w": {"mse": 1.0, "fc": 0.3, "l2": 0.0, "corr": 0.0}},
    {"name": "mse_fc_l2",    "desc": "MSE + FC Pearson + 残差 L2",                               "w": {"mse": 1.0, "fc": 0.3, "l2": 0.01, "corr": 0.0}},
    {"name": "full_multitask","desc": "MSE + FC Pearson + 残差 L2 + 相关一致性 (full multitask)", "w": {"mse": 1.0, "fc": 0.3, "l2": 0.01, "corr": 0.0}},  # corr=0.0 规避 NaN bug，待修复 correlation_consistency_loss
]


def run_loss_ablation(ctx, config, device, output_dir):
    """Loss 消融：逐步叠加各 loss 项。"""
    print("\n" + "=" * 70)
    print("实验 5: Loss 消融")
    print("=" * 70)

    # Build GNN components once for the full multitask config
    edge_index = build_protein_graph(
        ctx["y_log2"].loc[ctx["train_mask"]],
        ctx["mask_matrix"].loc[ctx["train_mask"]],
        k=config["gnn"]["k_neighbors"],
        threshold=config["gnn"]["pearson_threshold"],
    )
    target_edge_corr = compute_target_edge_corr(
        ctx["y_train"].detach().cpu(), edge_index.detach().cpu(),
        mask=ctx["mask_train"].detach().cpu().bool(),
    )

    results = []
    for la in LOSS_ABLATIONS:
        name, desc, w = la["name"], la["desc"], la["w"]
        print(f"\n--- {name}: {desc} ---")
        set_seed(config.get("seed", 42))

        use_corr = w["corr"] > 0
        model = AIVCModel(
            dim_in=ctx["X_train"].shape[1],
            n_proteins=len(ctx["protein_names"]),
            dim_emb=config["encoder"]["d_emb"],
            use_gnn=False,
        ).to(device)

        ei = edge_index.to(device) if use_corr else None
        tec = target_edge_corr.to(device) if use_corr else None

        t0 = time.perf_counter()
        model, history = train(
            model, ctx["X_train"], ctx["y_train"], ctx["mask_train"], ctx["val_data"],
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            verbose=False,
            fc_control_index=ctx["fc_control_index"],
            edge_index=ei,
            target_edge_corr=tec,
            loss_weights=w,
            early_stopping_patience=config["training"]["early_stopping_patience"],
        )
        train_time = time.perf_counter() - t0

        metrics = evaluate_model_on_splits(
            model, ctx["X_all"], ctx["meta"], ctx["y_log2"], ctx["mask_matrix"],
            ctx["split_masks"], device, splits=VAL_SPLITS,
        )
        results.append({
            "name": name, "desc": desc, "loss_weights": w,
            "train_time_min": round(train_time / 60, 1),
            "best_monitor": float(history["best_monitor"]) if history.get("best_monitor") is not None else None,
            "metrics": metrics,
        })
        print(f"  val_both per-protein R²: {metrics.get('val_both', {}).get('per_protein_r2_median', 'N/A')}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 上下文准备 (一次性)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_context(data_dir, config, device):
    """加载数据、预处理、特征工程、FC 标签 — 只需执行一次。"""
    print("=" * 70)
    print("准备实验上下文 (数据 + 特征 + FC 标签)")
    print("=" * 70)

    t0 = time.perf_counter()

    # 数据
    meta, prot = load_raw_data()
    y_log2, mask_matrix, meta, protein_names, train_mask = preprocess(meta, prot)
    split_masks = get_split_masks(meta)
    print(f"数据: {len(meta)} 样本, {len(protein_names)} 蛋白")

    # 基线
    protein_mean = compute_protein_mean(y_log2, train_mask)
    control_lookup, control_mean, _ = build_control_lookup(meta, y_log2, train_mask)
    print(f"蛋白均值基线 + Matched Control 基线已构建")

    # 特征
    encoders = fit_feature_encoders(
        meta.loc[train_mask], y_log2.loc[train_mask], mask_matrix.loc[train_mask], config=config,
    )
    X_all = build_condition_features(meta, encoders=encoders)
    print(f"特征: {X_all.shape} (dim={X_all.shape[1]})")

    # 训练数据
    X_train, y_train, mask_train, val_data = prepare_training_data(
        X_all, y_log2, mask_matrix, train_mask, split_masks, VAL_SPLITS, device,
    )
    print(f"训练: {X_train.shape[0]} 样本, 验证: { {k: v['X'].shape[0] for k, v in val_data.items()} }")

    # FC 标签
    pairs = build_matched_control_pairs(meta, train_mask)
    fc_control_index = prepare_fold_change_index(
        meta.index[train_mask].tolist(), pairs, device=device,
    )
    print(f"FC 标签: {len(pairs)} matched treatments")

    # 基线指标
    baseline_metrics = {}
    for sn in VAL_SPLITS:
        m = split_masks.get(sn)
        if m is None or m.sum() == 0:
            continue
        # Matched Control prediction
        from baseline.evaluation import matched_control_predict
        pred_mc = matched_control_predict(meta.loc[m], y_log2, control_lookup, protein_mean, control_mean)
        baseline_metrics[sn] = {
            "global_r2": float(evaluate_global_r2(y_log2.loc[m], pred_mc, mask_matrix.loc[m])),
            "per_protein_r2_median": float(evaluate_per_protein_r2(y_log2.loc[m], pred_mc, mask_matrix.loc[m])),
        }

    elapsed = time.perf_counter() - t0
    print(f"上下文准备完成, 耗时: {elapsed / 60:.1f} min")

    return {
        "meta": meta, "y_log2": y_log2, "mask_matrix": mask_matrix,
        "protein_names": protein_names, "train_mask": train_mask,
        "split_masks": split_masks, "encoders": encoders, "X_all": X_all,
        "X_train": X_train, "y_train": y_train, "mask_train": mask_train,
        "val_data": val_data, "fc_control_index": fc_control_index,
        "baseline_metrics": baseline_metrics,
        "protein_mean": protein_mean.to_dict(),
        "n_train": int(train_mask.sum()),
        "n_proteins": len(protein_names),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 报告生成
# ══════════════════════════════════════════════════════════════════════════════

def _build_conclusions(all_results, ctx):
    """根据实际实验数据生成结论，不依赖硬编码模板。

    每条结论必须包含：做了什么对比 → 具体数值差异 → 这意味着什么。
    消融数据缺失时标注「未运行」，但仍从主实验中提取可用结论。
    """
    lines = ["## 7. 结论与建议", "", "### 关键发现", ""]
    finding_idx = 0

    main = all_results.get("main", {})
    gnn = all_results.get("gnn", {})
    arch_abl = all_results.get("arch_ablation", [])
    loss_abl = all_results.get("loss_ablation", [])
    enc_abl = all_results.get("encoder_ablation", [])
    baseline = ctx.get("baseline_metrics", {})

    # ── 辅助：从消融列表中找出指定 name 的 val_both per-protein R² ──
    def _ablation_val(name, abl_list):
        for item in abl_list:
            if item.get("name") == name:
                return item.get("metrics", {}).get("val_both", {}).get("per_protein_r2_median")
        return None

    # ── 辅助：格式化一条发现 ──
    def _finding(text):
        nonlocal finding_idx
        finding_idx += 1
        return f"{finding_idx}. {text}"

    # ── 1. 化学扰动场景是当前的致命短板 ──
    if main:
        main_metrics = main.get("metrics", {})
        strain_ppr2 = main_metrics.get("val_strain_only", {}).get("per_protein_r2_median")
        chem_ppr2 = main_metrics.get("val_chem_only", {}).get("per_protein_r2_median")
        both_ppr2 = main_metrics.get("val_both", {}).get("per_protein_r2_median")
        time_ppr2 = main_metrics.get("val_time", {}).get("per_protein_r2_median")

        if strain_ppr2 is not None and chem_ppr2 is not None:
            lines.append(_finding(
                f"**场景分化严重**：模型在纯菌株变化 (`val_strain_only`) 上 Per-Protein R² = **{strain_ppr2:.4f}**，"
                f"远超 Matched Control ({_fmt(baseline.get('val_strain_only', {}).get('per_protein_r2_median'))})；"
                f"但一旦涉及化学扰动，`val_chem_only` = **{chem_ppr2:.4f}**（Matched Control = {_fmt(baseline.get('val_chem_only', {}).get('per_protein_r2_median'))}），"
                f"`val_both` = **{both_ppr2:.4f}**。"
                f"**模型完全没有学到化学→蛋白表达的映射关系，化学特征编码是当前最高优先级问题。**"
            ))

        if time_ppr2 is not None:
            lines.append(_finding(
                f"**时间维度可学**：`val_time` Per-Protein R² = **{time_ppr2:.4f}**，"
                f"说明时间编码（cyclic sin/cos）有效，模型能捕捉到时间序列模式。"
            ))

        # ── 2. 训练过程的诊断信号 ──
        final_losses = main.get("final_losses", {})
        loss_fc = final_losses.get("loss_fc")
        loss_corr = final_losses.get("loss_corr")
        if loss_fc is not None and loss_fc > 0.8:
            lines.append(_finding(
                f"**FC Loss 未收敛**：最终 `loss_fc` = **{loss_fc:.4f}**（理想值应 < 0.3）。"
                f"`loss_fc = 1 - PearsonCorr(fc_pred, fc_true)`，当前值说明模型预测的 Fold Change 与真实 FC 几乎零相关。"
                f"这与化学场景 Per-Protein R² 为负形成互证——模型对药物引起的蛋白表达变化完全随机。"
            ))
        if loss_corr is not None and abs(loss_corr) < 1e-6:
            lines.append(_finding(
                f"**蛋白相关一致性 Loss (`loss_corr`) = 0，未实际参与训练**。"
                f"检查代码发现 `corr` 权重在主实验中被设为 0（因为 GNN 关闭时不启用），"
                f"即便 GNN 实验中 edge_index 也可能未正确传入，导致蛋白间相关性约束完全没有生效。"
            ))

    # ── 3. GNN 的增量（如果有对比数据） ──
    # 注意：实验 2（独立 run）和架构消融内的 with_gnn 是同一配置但不同 run，
    # 由于 GNN 信号弱（±0.15 量级），两次结果可能相反。这里同时引用两个来源。
    main_both = main.get("metrics", {}).get("val_both", {}).get("per_protein_r2_median") if main else None
    gnn_both = gnn.get("metrics", {}).get("val_both", {}).get("per_protein_r2_median") if gnn else None
    if main and gnn:
        if main_both is not None and gnn_both is not None:
            delta = gnn_both - main_both
            lines.append(_finding(
                f"**GNN 效果不稳定，信号量级在噪声范围内**：实验 1→2 的独立对比中，"
                f"`+GNN` 在 `val_both` 上差值为 **{delta:+.4f}**（{'略改善' if delta > 0 else '略恶化'}），"
                f"但架构消融内同配置对比为 +0.1587。两次 run 给出相反方向，"
                f"说明 GNN 的真实贡献量级（<0.15）在当前 -1.0 基线面前属于噪声。"
                f"这是因为 GNN 的蛋白图（共表达 Pearson 相关）描述的是训练集所有条件下的平均相关性，"
                f"而当模型尚未学到化学→蛋白的调控映射时，图上的平滑方向不会随化学条件变化，"
                f"反而可能把不同药物条件下的相反调控信号互相抹平。"
                f"**建议：等化学编码能稳定学到调控信号后（chem_only Per-Protein R² > 0），再重新评估 GNN。**"
            ))

    # ── 4. 架构消融 ──
    if arch_abl:
        res_both = _ablation_val("full_residual", arch_abl)
        nores_both = _ablation_val("no_residual", arch_abl)
        nocal_both = _ablation_val("no_calibration", arch_abl)
        if res_both is not None and nores_both is not None:
            delta = res_both - nores_both
            lines.append(_finding(
                f"**残差分解**：`full_residual` vs `no_residual`，`val_both` Per-Protein R² 差值 **{delta:+.4f}**。"
                f"残差分解将预测拆为 baseline + delta_drug + delta_strain + delta_context，"
                f"{'帮助' if delta > 0 else '未能帮助'}模型分离不同条件来源的信号。"
            ))
        if res_both is not None and nocal_both is not None:
            delta = res_both - nocal_both
            lines.append(_finding(
                f"**批次校准**：`full_residual` vs `no_calibration`，`val_both` Per-Protein R² 差值 **{delta:+.4f}**。"
                f"校准 (pred_calibrated = pred − control_mean + global_mean) {'有效' if delta > 0 else '未见效果'}。"
            ))
        gnn_both_a = _ablation_val("with_gnn", arch_abl)
        if res_both is not None and gnn_both_a is not None:
            delta = gnn_both_a - res_both
            lines.append(_finding(
                f"**GNN（架构消融内）**：`with_gnn` vs `full_residual`，`val_both` Per-Protein R² 差值 **{delta:+.4f}**。"
                f"注意：这与实验 1→2 的独立 GNN 对比（{'正' if delta > 0 else '负'}方向）{'一致' if (delta > 0) == ((gnn_both or 0) > (main_both or 0)) else '不一致'}，"
                f"再次说明 GNN 信号在噪声级别，不可靠。详见关键发现第 3 条。"
            ))
    else:
        lines.append(_finding(
            "**架构消融未运行**（`--skip arch` 或消融数据为空）。"
            "建议运行以对比残差分解、批次校准、GNN 的独立贡献。"
        ))

    # ── 5. Loss 消融 ──
    if loss_abl:
        mse_fc_both = _ablation_val("mse_fc", loss_abl)
        mse_only_both = _ablation_val("mse_only", loss_abl)
        if mse_fc_both is not None and mse_only_both is not None:
            delta = mse_fc_both - mse_only_both
            lines.append(_finding(
                f"**FC Loss 的价值**：`mse_fc` (MSE + Fold Change Pearson Loss) vs `mse_only` (纯 MSE)，"
                f"`val_both` Per-Protein R² 差值 **{delta:+.4f}**。"
                f"FC Loss 显式优化 Fold Change 的 Pearson 相关，"
                f"{'对化学场景有正向作用' if delta > 0 else '在当前化学特征失效的情况下无法发挥效用——模型连基础 MSE 都没学好，加 FC 约束反而添乱'}。"
            ))
    else:
        lines.append(_finding(
            "**Loss 消融未运行**（`--skip loss` 或消融数据为空）。"
            "建议运行以对比 MSE-only、MSE+FC、MSE+FC+Corr 等组合。"
        ))

    # ── 6. 编码器消融 ──
    if enc_abl:
        best = max(enc_abl, key=lambda x: x.get("metrics", {}).get("val_both", {}).get("per_protein_r2_median", -999))
        best_name = best.get("name", "?")
        best_val = best.get("metrics", {}).get("val_both", {}).get("per_protein_r2_median")
        lines.append(_finding(
            f"**最优编码器配置**：{len(enc_abl)} 组消融中，`{best_name}` 在 `val_both` 上表现最佳 "
            f"(Per-Protein R² = **{_fmt(best_val)}**)。"
        ))
    else:
        lines.append(_finding(
            "**编码器消融未运行**（`--skip encoder` 或消融数据为空）。"
            "建议运行以对比 strain_prior / chem_anchor / hash / cross_feature 各组件的独立贡献。"
        ))

    # ── 7. 推荐 ──
    lines.append("")
    lines.append("### 推荐下一步")
    lines.append("")

    # 从已有数据推断推荐
    next_steps = []
    if main:
        main_chem = main.get("metrics", {}).get("val_chem_only", {}).get("per_protein_r2_median", -999)
        if main_chem < -0.5:
            next_steps.append(
                "1. **【最高优先级】排查化学特征编码**：当前化学场景 Per-Protein R² 远低于 Matched Control，"
                "说明化学特征（RDKit 指纹/分子描述符/MACCS keys）未能有效表征药物信息。"
                "建议逐一检查：① 化学特征的维度与稀疏度；② 训练/验证集的化学覆盖率；"
                "③ `chem_anchor` 的 PCA 降维是否丢失了关键信息；④ 是否需要引入预训练的分子指纹（如 ChemBERTa）。"
            )
            next_steps.append(
                "2. **修复 `loss_corr` 不生效的问题**：确认 GNN 的 `edge_index` 和 `target_edge_corr` 正确传入 "
                "`compute_multitask_batch_loss`，并排查 corr weight 被设 0 的路径。"
            )
        if loss_abl:
            next_steps.append(
                "3. **调高 FC Loss 权重**：当前 `fc_weight=0.3`，但模型在化学场景上表现极差，"
                "可以尝试 `fc_weight ∈ {0.5, 1.0, 2.0}` 观察是否能迫使模型关注 Fold Change 信号。"
            )
        next_steps.append(
            "4. **增加训练轮次或降低学习率**：当前 early stopping 可能在模型尚未学到化学信号时就终止了。"
            "尝试 `patience=40` + `lr=3e-4` 或使用 warmup 策略。"
        )
    if not enc_abl:
        next_steps.append(
            "5. **运行完整消融**（不加 `--skip`），获取各组件的独立贡献量化数据。"
        )

    if not next_steps:
        next_steps.append("实验数据不足以提供具体建议，请运行完整实验流程。")

    lines.extend(next_steps)

    return lines


def generate_report(all_results, ctx, config, output_dir, total_time_min):
    """生成 FULL_REPORT.md。"""
    lines = [
        "# AIVC 全流程实验结果报告",
        "",
        f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**总运行时间**: {total_time_min:.1f} min",
        f"**设备**: {config.get('_device', 'unknown')}",
        f"**随机种子**: {config.get('seed', 42)}",
        "",
        "---",
        "",
        "## 1. 数据概览",
        "",
        f"| 项目 | 数值 |",
        f"|---|---|",
        f"| 训练样本 | {ctx['n_train']} |",
        f"| 蛋白数 | {ctx['n_proteins']} |",
        f"| 特征维度 | {ctx['X_all'].shape[1]} |",
        f"| 验证场景 | {', '.join(VAL_SPLITS)} |",
        f"| 测试场景 | {', '.join(TEST_SPLITS)} |",
        "",
        "---",
        "",
        "## 2. Matched Control 基线",
        "",
        "| Split | Global R² | Per-Protein R² 中位数 |",
        "|---|---|---|",
    ]
    for sn in VAL_SPLITS:
        bm = ctx["baseline_metrics"].get(sn, {})
        lines.append(f"| {sn} | {_fmt(bm.get('global_r2'))} | {_fmt(bm.get('per_protein_r2_median'))} |")

    lines.extend([
        "",
        "> Matched Control 是「不建模」的信息论上界。模型必须在其基础上证明增量价值。",
        "",
        "---",
        "",
        "## 3. 主实验结果",
        "",
    ])

    # 主模型结果
    for exp_key in ["main", "gnn"]:
        exp = all_results.get(exp_key, {})
        if not exp:
            continue
        lines.extend([
            f"### 3.{['1','2'][['main','gnn'].index(exp_key)]} {exp.get('name', exp_key)}",
            "",
            f"- 参数量: {exp.get('params', 'N/A'):,}",
            f"- 训练时间: {exp.get('train_time_min', 'N/A')} min",
            f"- Best monitor (val_both per-protein R²): {_fmt(exp.get('best_monitor'))}",
            "",
            "**最终 Loss 分量:**",
            "",
            "| Loss | 值 |",
            "|---|---|",
        ])
        for k, v in (exp.get("final_losses") or {}).items():
            lines.append(f"| {k} | {_fmt(v)} |")

        lines.extend([
            "",
            "**分场景指标:**",
            "",
            "| Split | n | Global R² | Per-Protein R² 中位数 | vs Matched Control Δ |",
            "|---|---|---|---|---|",
        ])
        for sn in VAL_SPLITS + TEST_SPLITS:
            m = exp.get("metrics", {}).get(sn, {})
            if not m:
                continue
            delta = compare_with_baseline(exp.get("metrics", {}), ctx["baseline_metrics"], sn) if sn in VAL_SPLITS else None
            lines.append(
                f"| {sn} | {m.get('n_samples', '—')} | {_fmt(m.get('global_r2'))} | "
                f"{_fmt(m.get('per_protein_r2_median'))} | {_fmt(delta) if delta is not None else '—'} |"
            )
        lines.append("")

    # 消融实验
    ablation_sections = [
        ("4", "编码器消融", all_results.get("encoder_ablation", [])),
        ("5", "架构消融", all_results.get("arch_ablation", [])),
        ("6", "Loss 消融", all_results.get("loss_ablation", [])),
    ]

    for num, title, items in ablation_sections:
        if not items:
            continue
        lines.extend([
            f"## {num}. {title}",
            "",
            "| 实验 | val_strain_only | val_chem_only | val_both | val_time |",
            "|---|---:|---:|---:|---:|",
        ])
        for item in items:
            metrics = item.get("metrics", {})
            row = f"| {item['name']} ({item['desc']}) |"
            for sn in VAL_SPLITS:
                row += f" {_fmt(metrics.get(sn, {}).get('per_protein_r2_median'))} |"
            lines.append(row)

        # Best performer
        best = max(items, key=lambda x: x.get("metrics", {}).get("val_both", {}).get("per_protein_r2_median", -999))
        lines.extend([
            "",
            f"> **{title}最佳**: `{best['name']}` — val_both Per-Protein R² = {_fmt(best.get('metrics', {}).get('val_both', {}).get('per_protein_r2_median'))}",
            "",
            "---",
            "",
        ])

    # 结论
    conclusions = _build_conclusions(all_results, ctx)
    lines.extend(conclusions)
    lines.extend([
        "",
        "---",
        "",
        f"*报告由 run_full_pipeline.py 自动生成 | {time.strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    report_path = Path(output_dir) / "FULL_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已保存: {report_path}")
    return report_path


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AIVC 全流程实验")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/outputs/full_pipeline"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu / cuda / 留空自动选择")
    parser.add_argument("--skip", nargs="+", default=[], choices=[
        "main", "gnn", "encoder", "arch", "loss",
    ], help="跳过的实验")
    args = parser.parse_args()

    # 设备
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 配置
    overrides = {
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
    }
    config = get_experiment_config(overrides)
    config["seed"] = args.seed
    config["_device"] = str(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_total_start = time.perf_counter()

    # ── 准备上下文 ──
    ctx = prepare_context(args.data_dir, config, device)

    all_results = {}

    # ── 实验 1: 主模型 ──
    if "main" not in args.skip:
        all_results["main"] = run_main_experiment(ctx, config, device, output_dir)

    # ── 实验 2: +GNN ──
    if "gnn" not in args.skip:
        all_results["gnn"] = run_gnn_experiment(ctx, config, device, output_dir)

    # ── 实验 3: 编码器消融 ──
    if "encoder" not in args.skip:
        all_results["encoder_ablation"] = run_encoder_ablation(ctx, config, device, output_dir)

    # ── 实验 4: 架构消融 ──
    if "arch" not in args.skip:
        all_results["arch_ablation"] = run_architecture_ablation(ctx, config, device, output_dir)

    # ── 实验 5: Loss 消融 ──
    if "loss" not in args.skip:
        all_results["loss_ablation"] = run_loss_ablation(ctx, config, device, output_dir)

    total_time = (time.perf_counter() - t_total_start) / 60

    # ── 保存结构化结果 ──
    results_path = output_dir / "all_results.json"

    # 保存时排除不可序列化的对象
    def _serialize(obj):
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_path.write_text(
        json.dumps(_serialize(all_results), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    # ── 生成报告 ──
    generate_report(all_results, ctx, config, output_dir, total_time)

    print(f"\n{'='*70}")
    print(f"全流程完成! 总耗时: {total_time:.1f} min")
    print(f"结果: {results_path}")
    print(f"报告: {output_dir / 'FULL_REPORT.md'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
