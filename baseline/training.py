"""
训练模块：mask-aware 损失函数 + MLP 训练循环

升级方向（设计文档 Phase 2+）:
  - 多目标 loss: mask-aware MSE + FC Pearson loss + 残差 L2 正则 + 蛋白相关一致性 loss
  - GroupKFold 交叉验证（按 strain×chemical 分组）
"""
import numpy as np
import torch


def mask_aware_mse(pred, y_filled, mask):
    """
    mask-aware MSE: 缺失值不贡献梯度

    参数:
        pred:      (N, P) 预测值
        y_filled:  (N, P) 真实值（NA 位置已填 0）
        mask:      (N, P) 观测 mask（1=有值, 0=缺失）
    """
    diff = (pred - y_filled) ** 2
    return (diff * mask).sum() / mask.sum()


def prepare_training_data(
    X_all, y_log2, mask_matrix, train_mask, split_masks, val_splits, device
):
    """
    准备训练和验证的 PyTorch 张量

    返回:
        X_train_t, y_train_t, mask_train_t:  训练集张量
        val_data:                            验证集字典 {split_name: {X, y_filled, mask_t, ...}}
    """
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
    for split_name in val_splits:
        m = split_masks[split_name]
        if m.sum() == 0:
            continue
        val_data[split_name] = {
            "X": torch.tensor(X_all[m.values], dtype=torch.float32).to(device),
            "y_filled": torch.tensor(
                y_log2.loc[m].fillna(0).values, dtype=torch.float32
            ).to(device),
            "mask_t": torch.tensor(
                mask_matrix.loc[m].values, dtype=torch.float32
            ).to(device),
        }

    return X_train_t, y_train_t, mask_train_t, val_data


def train(
    model,
    X_train_t,
    y_train_t,
    mask_train_t,
    val_data,
    epochs=100,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    device="cpu",
    verbose=True,
):
    """
    MLP 训练循环（Adam + ReduceLROnPlateau）

    返回:
        model:   已加载最佳权重的模型
        history: {"train_loss": [...], "val_loss": {split: [...]}}
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    N_TRAIN = X_train_t.shape[0]
    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": {sn: [] for sn in val_data}}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N_TRAIN)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, N_TRAIN, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            pred = model(X_train_t[idx])
            loss = mask_aware_mse(pred, y_train_t[idx], mask_train_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_train_loss)

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for sn, vd in val_data.items():
                pred = model(vd["X"])
                v_loss = mask_aware_mse(pred, vd["y_filled"], vd["mask_t"]).item()
                history["val_loss"][sn].append(v_loss)
                val_losses.append(v_loss)

        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 20 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"train loss={avg_train_loss:.4f} | "
                f"val loss={avg_val_loss:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    # 加载最佳模型
    model.load_state_dict(best_state)
    if verbose:
        print(f"训练完成, best val loss={best_val_loss:.4f}")

    return model, history
