# Person A — Encoder 消融实验报告

- 日期：2026-08-07
- 分支：`person-a`
- 任务：A5（消融框架）+ A6（Encoder 消融与文档）
- 数据：13,412 个样本，预处理后 4,422 个蛋白；训练集 5,920 个样本
- 模型：Phase 1 `ConditionMLP`（所有实验参数量均为 1,269,062）
- 训练配置：40 epochs，batch size 256，Adam，lr=1e-3，weight decay=1e-5，seed=42，CPU
- 特征纪律：所有类别映射、统计锚点、PCA、标准化和 256 维投影仅使用 train 拟合
- 完整运行墙钟时间：约 140 秒（16 CPU threads）

运行命令：

```powershell
python -m experiments.run_encoder_ablation --epochs 40 --output-dir experiments/outputs
```

原始结果保存在本地忽略目录：

```text
experiments/outputs/encoder_ablation_results.json
```

## 结果

### Global R²

| 配置 | raw dim | best val loss | val_strain_only | val_chem_only | val_both | val_time |
|---|---:|---:|---:|---:|---:|---:|
| full | 302 | 0.3929 | **0.9290** | 0.9412 | 0.9322 | **0.9491** |
| no_strain_prior | 270 | 0.4000 | 0.9237 | **0.9461** | 0.9260 | 0.9465 |
| no_chem_anchor | 238 | 0.4407 | 0.9025 | 0.9354 | 0.9186 | 0.9390 |
| no_hash | 254 | 0.4297 | 0.9029 | 0.9398 | 0.9127 | 0.9414 |
| no_cross_features | 200 | 0.3948 | 0.9204 | 0.9449 | **0.9345** | 0.9467 |

### Per-Protein R² median

| 配置 | val_strain_only | val_chem_only | val_both | val_time |
|---|---:|---:|---:|---:|
| full | 0.2170 | 0.4451 | 0.3601 | **0.3785** |
| no_strain_prior | 0.1283 | 0.4854 | 0.2814 | 0.3606 |
| no_chem_anchor | 0.0254 | 0.4195 | 0.2298 | 0.2715 |
| no_hash | -0.0766 | 0.4669 | 0.0922 | 0.2987 |
| no_cross_features | **0.2304** | **0.5134** | **0.3870** | 0.3669 |

## 相对 full 的主要变化

- 去掉 strain prior：`val_strain_only` Per-Protein median 下降 0.0887，`val_both` 下降 0.0787，说明菌株统计先验对菌株外推有效。
- 去掉 chemical anchor：四个验证桶 Global R² 均下降；`val_strain_only` Per-Protein median 下降 0.1916，`val_both` 下降 0.1304，统计化合物锚点提供了稳定增益。
- 去掉 hash：`val_strain_only` Per-Protein median 从 0.2170 降到 -0.0766，`val_both` 从 0.3601 降到 0.0922，是本轮消融中最明显的退化，说明 hash 对 unseen 实体/批次泛化重要。
- 去掉 cross features：部分指标反而略升，尤其 `val_chem_only` 和 `val_both` 的 Per-Protein median；当前交叉 hash 特征没有在单 seed 实验中证明稳定增益，建议后续做多 seed 复验或缩小交叉维度。

## 结论

1. 建议保留 strain prior、chemical anchor 和 hash 特征。
2. cross features 暂列为待验证项，不应仅凭本次单 seed 结果宣称有效。
3. 所有配置最终输入均固定为 256 维，模型参数量完全一致，因此对比主要反映输入特征组的影响。
4. 本报告是 Person A 的参考 Encoder 消融；最终统一对标仍应使用 Person C 的共享训练循环和统一 FC PCC 指标复核。
