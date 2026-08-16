"""今天所有实验的统一 runner：13 配置 × 3 seed，4 worker 并行。

配置（主演进表 + 蛋白先验消融 + 菌株先验消融 + 拆分 A/B），每个配置跑 3 seed，
输出完整指标（val + test 各 split 的 sample_corr / fc_pcc / ctx_res / drug_res / ppr2），
并存权重。Linux 下 multiprocessing 用 fork，子进程通过模块级 _SHARED 共享数据。

用法:
    python -m experiments.run_all_today --workers 4 --device cuda
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.data import load_raw_data, preprocess, get_split_masks
from baseline.evaluation import (
    build_control_lookup,
    build_matched_control_pairs,
    build_train_residual_means,
    compute_fold_change,
    evaluate_official_metrics,
)
from baseline.model import ConditionMLP, ControlTreatSplitModel, SplitConditionMLP
from aivc.training import (
    build_residual_mean_tensors,
    prepare_fold_change_index,
    prepare_training_data,
    train,
)
from aivc.entity_representations import StrainGenomeEncoder, compute_strain_bias
from aivc.external_protein import ProteinPriorEncoder
from experiments.ablation_encoder import prepare_ablation_features

SEEDS = [42, 123, 2024]
VAL_SPLITS = ["val_strain_only", "val_chem_only", "val_both", "val_time"]
TEST_SPLITS = ["test_strain_only", "test_chem_only", "test_both", "test_time"]
OUT_DIR = Path("experiments/outputs/today")
LOWRANK_K = 20

# 上下文 vs 化药+菌株 的语义分组（拆分 A 用）
CTX_GROUPS = ["media", "instruments", "temperature", "time", "hash_plate"]
BIO_GROUPS = [
    "strains", "chemicals", "strain_prior", "chem_anchor",
    "hash_chemical", "chemical_structure", "cross_features",
]

_SHARED = None


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_protein_prior(kind, protein_names, encoder):
    """构建蛋白先验矩阵 (n_protein, d)。kind=None/'bias' 返回 None。"""
    if kind in (None, "none", "bias"):
        return None
    if kind == "esm":
        return encoder.embed(protein_names)
    if kind == "shuffle":
        e = encoder.embed(protein_names)
        rng = np.random.default_rng(7)
        return e[rng.permutation(len(e))]
    if kind == "go":
        return encoder.go_features(protein_names)
    if kind == "esm_go":
        return np.concatenate(
            [encoder.embed(protein_names), encoder.go_features(protein_names)], axis=1
        )
    raise ValueError(f"unknown protein prior kind: {kind}")


def _build_lowrank_target(meta, y_log2, mask_matrix, train_mask, pairs):
    """生成低秩去噪 FC 目标 (n_train, n_protein)。"""
    fc = compute_fold_change(meta, y_log2, mask_matrix, train_mask, pairs)
    fc_true = fc["fc_true"].to_numpy(dtype=np.float64)
    fc_mask = fc["fc_mask"].to_numpy(dtype=bool)
    filled = np.nan_to_num(fc_true, nan=0.0)
    col_cnt = fc_mask.sum(axis=0).astype(np.float64)
    col_mean = np.divide(
        np.where(fc_mask, fc_true, 0.0).sum(axis=0), col_cnt,
        out=np.zeros(fc_true.shape[1]), where=col_cnt > 0,
    )
    centered = filled - col_mean[None, :]
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    recon = (U[:, :LOWRANK_K] * s[:LOWRANK_K]) @ Vt[:LOWRANK_K, :] + col_mean[None, :]
    recon = np.where(fc_mask, recon, np.nan)

    train_sids = list(meta.index[train_mask])
    pos = {sid: i for i, sid in enumerate(train_sids)}
    target = np.full((len(train_sids), fc_true.shape[1]), np.nan, dtype=np.float32)
    for i, sid in enumerate(fc["fc_true"].index):
        if sid in pos:
            target[pos[sid]] = recon[i].astype(np.float32)
    return target


def _build_split_features(meta, y_log2, mask_matrix, train_mask):
    """构建拆分 A 的 X_ctx / X_bio（raw 特征按语义拆分后再投影）。"""
    from baseline.features import build_raw_condition_features, fit_feature_encoders

    encoders = fit_feature_encoders(
        meta.loc[train_mask], y_log2.loc[train_mask], mask_matrix.loc[train_mask]
    )
    raw = build_raw_condition_features(meta, encoders)  # (N, raw_dim)
    slices = encoders["feature_slices"]  # {group: (start, end)}

    def _concat(groups):
        cols = []
        for g in groups:
            if g in slices:
                start, end = slices[g]
                cols.append(raw[:, start:end])
        return np.concatenate(cols, axis=1).astype(np.float32)

    x_ctx = _concat(CTX_GROUPS)
    x_bio = _concat(BIO_GROUPS)
    return x_ctx, x_bio


def _load_shared():
    """加载数据 + 特征 + 先验，返回共享 dict（fork 继承）。"""
    meta, prot = load_raw_data()
    y_log2, mask_matrix, meta, protein_names, _ = preprocess(meta, prot)
    split_masks = get_split_masks(meta)
    train_mask = meta["split_final"].astype(str).eq("train")

    features = prepare_ablation_features(meta, y_log2, mask_matrix)
    X_full = features["full"]["X"]
    X_no_chem = features["no_chemical_structure"]["X"]
    X_no_strain = features["no_strain_prior"]["X"]

    prot_enc = ProteinPriorEncoder()

    strain_bias, mu_0 = compute_strain_bias(meta, y_log2, mask_matrix, train_mask)
    strain_bias_all = np.stack([
        strain_bias.get(s, np.zeros(y_log2.shape[1], dtype=np.float32))
        for s in meta["Strains"].astype(str)
    ])

    genome_enc = StrainGenomeEncoder()
    genome_all = genome_enc.transform(meta)

    control_lookup, _, _ = build_control_lookup(meta, y_log2, train_mask=None)
    train_stats = build_train_residual_means(meta, y_log2, mask_matrix, train_mask)
    pairs = build_matched_control_pairs(meta, train_mask)

    lowrank_target = _build_lowrank_target(meta, y_log2, mask_matrix, train_mask, pairs)
    x_ctx, x_bio = _build_split_features(meta, y_log2, mask_matrix, train_mask)
    x_split = np.concatenate([x_ctx, x_bio], axis=1).astype(np.float32)

    return {
        "meta": meta, "y_log2": y_log2, "mask_matrix": mask_matrix,
        "protein_names": protein_names, "train_mask": train_mask,
        "split_masks": split_masks,
        "X_full": X_full, "X_no_chem": X_no_chem, "X_no_strain": X_no_strain,
        "X_ctx": x_ctx, "X_bio": x_bio, "X_split": x_split,
        "prot_enc": prot_enc, "strain_bias_all": strain_bias_all, "mu_0": mu_0,
        "genome_all": genome_all,
        "control_lookup": control_lookup, "train_stats": train_stats, "pairs": pairs,
        "lowrank_target": lowrank_target,
    }


def _pick_X(config, shared):
    if config["feat"] == "no_chemical_structure":
        return shared["X_no_chem"]
    if config["feat"] == "no_strain_prior":
        base = shared["X_no_strain"]
    else:
        base = shared["X_full"]
    if config.get("genome", False):
        base = np.concatenate([base, shared["genome_all"]], axis=1).astype(np.float32)
    return base


def _predict_all(model, X, device, batch_size=2048):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            x = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            out = model(x)
            if isinstance(out, dict):
                out = out["y_pred"]
            preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def _predict_split_b(model, shared, X, device):
    """拆分 B 预测：对照样本 = ctrl，处理样本 = ctrl + treat。"""
    meta = shared["meta"]
    pert = meta["perturbation_no_concentration"].astype(str).str.strip().str.lower()
    is_ctrl = pert.isin(["water", "dmso"]).to_numpy(dtype=bool)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, X.shape[0], 2048):
            x = torch.as_tensor(X[start:start + 2048], dtype=torch.float32, device=device)
            ctrl = model.forward_ctrl(x)
            treat = model.forward_treat(x)
            m = torch.as_tensor(is_ctrl[start:start + 2048], dtype=torch.bool, device=device)
            out = torch.where(m.unsqueeze(1), ctrl, treat)
            preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def _evaluate(model, shared, X, device, add_strain_bias=False, predict_fn=None):
    meta = shared["meta"]
    y_log2 = shared["y_log2"]
    mask = shared["mask_matrix"]
    split_masks = shared["split_masks"]
    control_lookup = shared["control_lookup"]
    train_stats = shared["train_stats"]

    pred_all = predict_fn(model, shared, X, device) if predict_fn else _predict_all(model, X, device)
    if add_strain_bias:
        pred_all = pred_all + shared["strain_bias_all"]
    pred_df = pd.DataFrame(pred_all, index=meta.index, columns=y_log2.columns)

    metrics = {}
    for split_name in VAL_SPLITS + TEST_SPLITS:
        m = split_masks.get(split_name)
        if m is None or int(m.sum()) == 0:
            continue
        metrics[split_name] = evaluate_official_metrics(
            meta, y_log2, mask, pred_df, m, control_lookup, train_stats
        )
    return metrics


def _train_standard(shared, X, device, config, seed):
    """标准 ConditionMLP 训练（可选蛋白先验 / 菌株偏置输出端 / 低秩目标）。"""
    meta = shared["meta"]
    y_log2 = shared["y_log2"]
    mask = shared["mask_matrix"]
    train_mask = shared["train_mask"]
    split_masks = shared["split_masks"]
    protein_names = shared["protein_names"]

    E = _build_protein_prior(config["prior"], protein_names, shared["prot_enc"])
    use_bias = config["prior"] == "bias"
    E_t = torch.as_tensor(E, dtype=torch.float32) if E is not None else None

    model = ConditionMLP(
        X.shape[1], len(protein_names), hidden=256, dropout=0.1,
        protein_prior=E_t, use_bias=use_bias,
    ).to(device)

    x_train, y_train, mask_train, val_data = prepare_training_data(
        X, y_log2, mask, train_mask, split_masks, VAL_SPLITS, device
    )
    use_strain_out = config.get("strain_out", False)
    if use_strain_out:
        sb = torch.as_tensor(shared["strain_bias_all"], dtype=torch.float32, device=device)
        y_train = y_train - sb[train_mask.values]

    fc_control_index = None
    ctx_t = None
    drug_t = None
    loss_weights = None
    denoised_t = None
    if config["loss"] != "mse":
        fc_control_index = prepare_fold_change_index(
            meta.index[train_mask].tolist(), shared["pairs"], device=device
        )
        ctx_t, drug_t = build_residual_mean_tensors(
            meta, y_log2, mask, train_mask, device=device
        )
        loss_weights = {"mse": 1.0, "fc": 1.0, "ctx": 0.5, "drug": 0.5, "l2": 0.01, "corr": 0.0}
    if config.get("lowrank", False):
        denoised_t = torch.as_tensor(shared["lowrank_target"], dtype=torch.float32, device=device)

    model, history = train(
        model, x_train, y_train, mask_train, val_data,
        epochs=100, batch_size=256, lr=1e-3, weight_decay=1e-5,
        device=device, verbose=False,
        fc_control_index=fc_control_index, ctx_mean_t=ctx_t, drug_mean_t=drug_t,
        loss_weights=loss_weights, denoised_fc_target=denoised_t,
    )
    return model, use_strain_out


def _train_split_a(shared, device, config, seed):
    """拆分 A：SplitConditionMLP 在拼接特征 [ctx|bio] 上训练。"""
    meta = shared["meta"]
    y_log2 = shared["y_log2"]
    mask = shared["mask_matrix"]
    train_mask = shared["train_mask"]
    split_masks = shared["split_masks"]
    protein_names = shared["protein_names"]
    X = shared["X_split"]

    E = _build_protein_prior(config["prior"], protein_names, shared["prot_enc"])
    E_t = torch.as_tensor(E, dtype=torch.float32) if E is not None else None

    ctx_dim = shared["X_ctx"].shape[1]
    bio_dim = shared["X_bio"].shape[1]
    model = SplitConditionMLP(ctx_dim, bio_dim, len(protein_names), hidden=256, dropout=0.1).to(device)

    x_train, y_train, mask_train, val_data = prepare_training_data(
        X, y_log2, mask, train_mask, split_masks, VAL_SPLITS, device
    )
    if config.get("strain_out", False):
        sb = torch.as_tensor(shared["strain_bias_all"], dtype=torch.float32, device=device)
        y_train = y_train - sb[train_mask.values]

    fc_control_index = prepare_fold_change_index(
        meta.index[train_mask].tolist(), shared["pairs"], device=device
    )
    ctx_t, drug_t = build_residual_mean_tensors(meta, y_log2, mask, train_mask, device=device)
    loss_weights = {"mse": 1.0, "fc": 1.0, "ctx": 0.5, "drug": 0.5, "l2": 0.01, "corr": 0.0}
    denoised_t = torch.as_tensor(shared["lowrank_target"], dtype=torch.float32, device=device)

    model, history = train(
        model, x_train, y_train, mask_train, val_data,
        epochs=100, batch_size=256, lr=1e-3, weight_decay=1e-5,
        device=device, verbose=False,
        fc_control_index=fc_control_index, ctx_mean_t=ctx_t, drug_mean_t=drug_t,
        loss_weights=loss_weights, denoised_fc_target=denoised_t,
    )
    return model, config.get("strain_out", False)


def _run_one(task):
    config_name, seed = task
    device = torch.device(_DEVICE)
    _set_seed(seed)
    config = CONFIGS[config_name]

    if config["model"] == "split_a":
        model, use_strain_out = _train_split_a(_SHARED, device, config, seed)
        X = _SHARED["X_split"]
    elif config["model"] == "split_b":
        model, use_strain_out = _train_split_b(_SHARED, device, config, seed)
        X = _pick_X(config, _SHARED)
    else:
        X = _pick_X(config, _SHARED)
        model, use_strain_out = _train_standard(_SHARED, X, device, config, seed)

    predict_fn = _predict_split_b if config["model"] == "split_b" else None
    metrics = _evaluate(model, _SHARED, X, device, add_strain_bias=use_strain_out, predict_fn=predict_fn)

    # 存权重
    weight_path = OUT_DIR / "weights" / f"{config_name}_seed{seed}.pt"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weight_path)

    return config_name, seed, metrics


def _train_split_b(shared, device, config, seed):
    """拆分 B：对照模型学基线 + 扰动模型学 Δ。"""
    meta = shared["meta"]
    y_log2 = shared["y_log2"]
    mask = shared["mask_matrix"]
    train_mask = shared["train_mask"]
    split_masks = shared["split_masks"]
    protein_names = shared["protein_names"]
    X = _pick_X(config, shared)

    fc = compute_fold_change(meta, y_log2, mask, train_mask, shared["pairs"])
    fc_true = fc["fc_true"]
    fc_mask = fc["fc_mask"]

    n = len(meta)
    p = len(protein_names)
    delta_all = np.zeros((n, p), dtype=np.float32)
    delta_mask_all = np.zeros((n, p), dtype=bool)
    for sid in fc_true.index:
        i = meta.index.get_loc(sid)
        delta_all[i] = np.nan_to_num(fc_true.loc[sid].to_numpy(dtype=np.float32), nan=0.0)
        delta_mask_all[i] = fc_mask.loc[sid].to_numpy(dtype=bool)

    # 基线标签：对照=自己 y，处理=matched control = y - Δ
    baseline_all = np.nan_to_num(y_log2.to_numpy(dtype=np.float32), nan=0.0)
    for sid in fc_true.index:
        i = meta.index.get_loc(sid)
        baseline_all[i] = baseline_all[i] - delta_all[i]

    device_t = device
    model = ControlTreatSplitModel(X.shape[1], p, hidden=256, dropout=0.1).to(device_t)

    # 训练 ctrl_net（基线标签）和 treat_net（Δ 标签），各自用 mask-aware MSE
    x_train = torch.as_tensor(X[train_mask.values], dtype=torch.float32, device=device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    y_base = torch.as_tensor(baseline_all[train_mask.values], dtype=torch.float32, device=device_t)
    y_delta = torch.as_tensor(delta_all[train_mask.values], dtype=torch.float32, device=device_t)
    m_base = torch.as_tensor(mask.to_numpy(dtype=bool)[train_mask.values], dtype=torch.float32, device=device_t)
    m_delta = torch.as_tensor(delta_mask_all[train_mask.values], dtype=torch.float32, device=device_t)

    for epoch in range(100):
        model.train()
        perm = torch.randperm(x_train.shape[0], device=device_t)
        for start in range(0, x_train.shape[0], 256):
            idx = perm[start:start + 256]
            xb = x_train[idx]
            base_pred = model.ctrl_net(xb)
            treat_pred = model.treat_net(xb)
            loss_base = (((base_pred - y_base[idx]) ** 2) * m_base[idx]).sum() / m_base[idx].sum().clamp_min(1)
            loss_treat = (((treat_pred - y_delta[idx]) ** 2) * m_delta[idx]).sum() / m_delta[idx].sum().clamp_min(1)
            loss = loss_base + loss_treat
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    # 拆分 B 的 ctrl_net 已学基线（含菌株特异），无需额外 strain_out 偏置
    return model, False


def main():
    global _SHARED, _DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    _DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(_DEVICE)
    print(f"设备={device}, workers={args.workers}")

    _SHARED = _load_shared()
    print("共享数据加载完成")

    tasks = [(name, seed) for name in CONFIGS for seed in SEEDS]
    print(f"共 {len(tasks)} 个训练任务")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.workers > 1 and _DEVICE.startswith("cuda"):
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            results = pool.map(_run_one, tasks)
    else:
        results = [_run_one(t) for t in tasks]

    # 汇总：{config: {split: {metric: [seed1, seed2, seed3]}}}
    summary = {}
    for config_name, seed, metrics in results:
        summary.setdefault(config_name, {})
        for split, m in metrics.items():
            summary[config_name].setdefault(split, {})
            for k, v in m.items():
                summary[config_name][split].setdefault(k, []).append(v)

    # 存原始结果
    raw_out = OUT_DIR / "raw_results.json"
    raw_out.write_text(json.dumps({f"{c}_seed{s}": m for c, s, m in results}, default=str, indent=2), encoding="utf-8")

    # 计算 mean±std
    agg = {}
    for config_name, splits in summary.items():
        agg[config_name] = {}
        for split, metrics in splits.items():
            agg[config_name][split] = {}
            for k, vals in metrics.items():
                arr = np.array([v for v in vals if v is not None and v == v], dtype=float)
                if arr.size:
                    agg[config_name][split][k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0}

    agg_out = OUT_DIR / "summary_mean_std.json"
    agg_out.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成，结果写入 {OUT_DIR}")


# 配置定义：13 个
CONFIGS = {
    "condmlp_original": dict(feat="no_chemical_structure", loss="mse", prior=None, model="condmlp"),
    "condmlp_full":     dict(feat="full", loss="residual", prior=None, model="condmlp"),
    "condmlp_lowrank":  dict(feat="full", loss="residual", prior=None, lowrank=True, model="condmlp"),
    "protein_bias":     dict(feat="full", loss="residual", prior="bias", lowrank=True, model="condmlp"),
    "protein_esm":      dict(feat="full", loss="residual", prior="esm", lowrank=True, model="condmlp"),
    "protein_shuffle":  dict(feat="full", loss="residual", prior="shuffle", lowrank=True, model="condmlp"),
    "protein_go":       dict(feat="full", loss="residual", prior="go", lowrank=True, model="condmlp"),
    "protein_esm_go":   dict(feat="full", loss="residual", prior="esm_go", lowrank=True, model="condmlp"),
    "strain_in":        dict(feat="full", loss="residual", prior="esm_go", lowrank=True, model="condmlp"),
    "strain_out":       dict(feat="no_strain_prior", loss="residual", prior="esm_go", lowrank=True, strain_out=True, model="condmlp"),
    "strain_genome":    dict(feat="no_strain_prior", loss="residual", prior="esm_go", lowrank=True, strain_out=True, genome=True, model="condmlp"),
    "split_a":          dict(feat="no_strain_prior", loss="residual", prior="esm_go", lowrank=True, strain_out=True, genome=True, model="split_a"),
    "split_b":          dict(feat="no_strain_prior", loss="residual", prior="esm_go", lowrank=True, strain_out=True, genome=True, model="split_b"),
}


if __name__ == "__main__":
    main()
