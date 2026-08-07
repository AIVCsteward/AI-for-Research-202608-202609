# Person A experiments

## Encoder 特征消融

### 仅生成特征

```powershell
python -m experiments.ablation_encoder --output-dir experiments/outputs
```

这会生成 5 组固定 256 维特征：`full`、`no_strain_prior`、`no_chem_anchor`、`no_hash`、`no_cross_features`。
生成的 `.npy` 和 manifest 只放在本地 `experiments/outputs/`，已被 `.gitignore` 忽略。

### 用当前 Phase 1 MLP 跑 A6 参考对比

```powershell
python -m experiments.run_encoder_ablation --epochs 40 --output-dir experiments/outputs
```

结果写入 `encoder_ablation_results.json`，默认评估四个验证桶：`val_strain_only`、`val_chem_only`、`val_both`、`val_time`。
这个 runner 是 Person B 的残差解码器接入前的参考实验；B/C 合并后应复用同一组 encoder 配置，并把各组 `X` 送入最终训练循环。

## 数据纪律

- 两个 `*_proteome_raw_*.csv` 已在根目录 `.gitignore` 中忽略，不能 `git add -f`。
- 所有统计表征、PCA、标准化和 256 维投影只用 `split_final == "train"` 拟合。
- 生成结果不要提交，提交代码、manifest 格式和文档即可。

## 已完成结果

2026-08-07 的 40 epoch 单 seed 消融结果与结论见 `experiments/encoder_ablation_report.md`。
