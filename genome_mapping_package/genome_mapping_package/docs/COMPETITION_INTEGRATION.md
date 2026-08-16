# GOAI 比赛接入说明

## 输入

脚本读取官方 metadata 的 `sample_ID`、`Strains` 和 `split_final`。菌株名称必须属于冻结索引中的 BAH、BAI、CEK、CGD、CRD、DHY210；未知名称会停止处理，不会静默回退。

## 输出

`competition_genome_features.npz` 包含：

- `sample_ids`：与输入 metadata 顺序一致；
- `strain_ids`：每个样本的比赛菌株代码；
- `genome_features`：形状为 `N×24` 的 float32 特征；
- `feature_valid_mask`：形状为 `N×24` 的逐维布尔掩码；
- `source_file_index`：样本来自 train_val 或 test metadata；
- `source_row_index`：样本在对应原始 metadata 中的零基行号。

`competition_sample_strain_mapping.csv` 保存可人工检查的 sample_ID、split、比赛菌株、公开菌株、映射状态、置信度及代理标记。

`competition_genome_manifest.json` 保存输入文件 SHA-256、样本数、split 数、特征定义哈希和输出文件哈希。

## 模型使用

模型应同时输入 `genome_features` 与 `feature_valid_mask`。不能把无效维度的数值零解释成真实生物学零值。DHY210 是 S288C 代理，只有来源明确的维度有效。

训练、验证和测试使用同一冻结转换；禁止使用验证或测试蛋白标签重新拟合标准化器、PCA 或缺失值统计。

