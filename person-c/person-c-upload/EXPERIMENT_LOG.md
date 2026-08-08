# EXPERIMENT LOG

## 当前状态

C5–C6 框架已经建立，尚未启动完整真实数据训练。前三组 loss 消融可进入训练准备；
`full_multitask` 等待 B4 交付修正后的最终 `edge_index`，随后由 C3 按边顺序计算
`target_edge_corr` 再接入训练。

## 数据纪律

- 所有编码器、归一化统计、FC 标签及相关性目标只能由 `split_final == "train"` 的行计算。
- 验证集只用于分场景评估和 `val_both` early stopping。
- 测试真值不得参与训练、特征拟合、阈值选择或模型选择。
- 所有提交预测保持 log2 尺度，并记录 `prediction_scale=log2`。

## Loss 消融计划

| 实验 | MSE | FC Pearson | 残差 L2 | 相关一致性 | 状态 |
|---|---:|---:|---:|---:|---|
| mse_only | 1.0 | 0.0 | 0.0 | 0.0 | ready |
| mse_fc | 1.0 | 0.3 | 0.0 | 0.0 | ready |
| mse_fc_l2 | 1.0 | 0.3 | 0.01 | 0.0 | ready |
| full_multitask | 1.0 | 0.3 | 0.01 | 0.1 | waiting_b4_edge_index |

## 正式运行后的记录要求

每组实验必须由 C5 结构化结果自动生成记录，包括随机种子、完整配置、最佳 epoch、
五项 loss 历史、四个验证场景指标、基线对标、checkpoint 路径和提交文件检查结果。
测试场景指标只有在训练和模型选择全部结束后才允许计算，并必须单独标注为自评结果。
