# GOAI 虚拟细胞 V2 接口合同

## 1. 合同状态

本文件定义各执行线程之间必须保持一致的命名、数据边界和模型输入输出。标为“冻结”的内容未经 Main 验收不得修改；标为“待审计”的内容不得由任一执行线程自行决定。

## 2. 路径与工件

```text
D:\虚拟细胞\WAYB_WAYC\                 # 真实比赛数据，不提交 Git
D:\虚拟细胞\references\                # 稳定参考资料
D:\虚拟细胞\project_v2\                # V2 合同与验收记录
D:\虚拟细胞\external_data\chemistry\  # 化学外部数据派生工件
D:\虚拟细胞\external_data\genome\     # 基因组外部数据派生工件
```

外部大型原始文件不得默认写入 Git 仓库；应保存 source manifest、下载脚本和 SHA-256。

## 3. 样本和字段命名

以下真实字段名冻结，不得擅自改成教程示例名：

| 角色 | 冻结字段名 |
|---|---|
| 样本唯一键 | `sample_ID` |
| 划分 | `split_final` |
| 菌株 | `Strains` |
| 化合物 | `perturbation_no_concentration` |
| 扰动编号 | `pert_id` |
| 培养基 | `Medium` |
| 温度 | `Temperature` |
| 处理时间 | `pert_time` |
| 时间单位 | `pert_time_unit` |
| 数据来源 | `data_source` |
| 仪器 | `instrument` |
| 实验板 | `Yeast_cell_plate` |
| 蛋白孔 | `protein_well` |
| 菌株角色 | `strain_role` |
| 化合物角色 | `chemical_role` |

内部变量名可以使用 `strain`、`chemical`、`medium` 等别名，但必须在一个明确的 schema adapter 中转换，不能在不同模块中重复、隐式重命名。

## 4. 划分名称

冻结划分：

- `train`
- `val_chem_only`
- `val_strain_only`
- `val_both`
- `val_time`
- `test_chem_only`
- `test_strain_only`
- `test_both`
- `test_time`

任何统计量拟合函数必须显式接收 `fit_sample_ids`，默认仅允许 `train`。

## 5. 蛋白特征合同

### 5.1 待审计项

- 教程报告的保留蛋白数：4,232。
- 当前真实文件按既有流程得到的保留蛋白数：4,422。
- 在数据合同审计通过前，不冻结具体数量，不允许强制裁剪。

### 5.2 冻结原则

- 蛋白列表只能由 train 标签计算。
- 列顺序写入 `project_v2/data_contract/feature_contract.json` 后冻结。
- 模型、损失、评估和提交均以该顺序为唯一依据。
- `n_proteins = len(feature_contract.proteins)`，禁止在代码中硬编码 4,232 或 4,422。

## 6. 目标与 mask

统一张量约定：

```text
y_true: float32 [batch, n_proteins], log2 scale
y_pred: float32 [batch, n_proteins], log2 scale
y_baseline: float32 [batch, n_proteins], log2 scale
delta_response: float32 [batch, n_proteins]
delta_batch: float32 [batch, n_proteins]
obs_mask: bool [batch, n_proteins]
```

缺失值加载后可以在张量中以安全值填充，但所有损失与指标必须同时使用 `obs_mask`，填充值本身不得贡献梯度或统计量。

## 7. Matched Control 与 FC

### 7.1 Control 匹配

正式匹配键由数据合同审计生成并写入 `control_matching_spec.json`。预期至少包含：

- Water/DMSO 官方映射；
- `Strains`；
- `Medium`；
- `Temperature`；
- `pert_time` 和 `pert_time_unit`；
- `data_source`；
- `instrument`；
- `Yeast_cell_plate`。

未经审计不得删除匹配字段以增加样本数量。

### 7.2 训练监督 FC

在 train 中构造：

```text
fc_true = y_treatment - y_control
fc_mask = treatment_obs_mask AND control_obs_mask
```

所有相减与 PCC/R² 只使用 `fc_mask=True` 的共同有效位置。

### 7.3 官方预测 FC

规划版规则定义：

```text
fc_pred_official = y_pred_treatment - y_control_observed
```

正式实现以审计后的官方口径为准。模型内部的 `delta_response` 是生物扰动分支，但未经验证不得自动等同于官方 `fc_pred_official`。训练和报告必须明确写出使用的是哪一种 FC。

## 8. 外部化学特征接口

化学模块应输出：

```text
chemical_id: string
raw_name: string
pubchem_cid: nullable string
inchikey: nullable string
canonical_smiles: nullable string
isomeric_smiles: nullable string
mapping_status: confirmed | proxy | special_control | unresolved
mapping_confidence: high | medium | low | none
morgan_fp: float32 [morgan_dim]
rdkit_descriptors: float32 [descriptor_dim]
feature_valid_mask: bool [chemical_feature_dim]
```

要求：

- `morgan_dim` 和 descriptor 列表写入 `feature_schema.json`，模型不得自行重算另一套顺序。
- descriptor 标准化器只在训练化合物上拟合。
- Water/DMSO/Quality Control 必须有明确特殊状态。
- unresolved 化合物使用显式 fallback，并保留合法性标记。
- shuffle 实验只打乱实体到特征的对应关系，不改变特征分布；随机种子必须记录。

## 9. 外部基因组特征接口

基因组模块应输出：

```text
strain_id: string
external_isolate_id: nullable string
mapping_type: exact | supported | proxy | unresolved
mapping_confidence: high | medium | low | none
genome_features: float32 [genome_dim]
feature_valid_mask: bool [genome_dim]
```

要求：

- BAH、BAI、CEK、CGD、CRD 的映射必须有证据记录。
- DHY210 使用 S288C 时固定为 `mapping_type=proxy`。
- 标准化和 PCA 只能在训练菌株上拟合。
- 验证/测试菌株只执行冻结转换。
- unresolved 菌株使用显式 fallback，不得伪造精确基因组。

## 10. 模型前向接口

建议统一返回字典：

```text
{
  "y_pred": y_pred,
  "y_baseline": y_baseline,
  "delta_response": delta_response,
  "delta_batch": delta_batch,
  "chemical_latent": optional,
  "genome_latent": optional,
  "similarity_gate": optional
}
```

必须满足：

```text
y_pred == y_baseline + delta_response + delta_batch
```

允许浮点误差，但必须有自动测试验证该恒等式。

## 11. 分支输入限制

| 模块 | 允许输入 | 禁止输入 |
|---|---|---|
| 基础状态 | genome、medium、temperature、time | chemical、batch |
| 化合物编码 | Morgan、descriptors、validity | protein labels、batch |
| 响应交互 | chemical、genome、medium、temperature、time | batch |
| 批次校正 | data_source、instrument、plate/well | chemical、genome、strain、medium、temperature、time |

这些限制必须由构造函数接口和测试保证，不能只写在文档中。

## 12. 相似药物响应接口

训练期可构建：

```text
train_chemical_fc_anchor: [n_train_chemicals, n_proteins]
train_chemical_fc_mask: [n_train_chemicals, n_proteins]
tanimoto_to_train: [n_samples_or_chemicals, n_train_chemicals]
```

所有 FC anchor 只能由 train 标签生成。对验证/测试化合物，只允许使用结构相似度查询训练 anchor。低相似度 fallback、邻居数量和门控规则写入配置并记录。

## 13. 损失接口

统一输出分项损失：

```text
{
  "loss_total": scalar,
  "loss_absolute": scalar,
  "loss_fc": scalar,
  "loss_dep": scalar,
  "loss_pathway": scalar,
  "loss_batch_reg": scalar
}
```

必须自动验证：

```text
loss_total == sum(weight_i * loss_i)
```

未启用的损失项权重为零，但仍应在日志中明确显示。

## 14. 评估与报告接口

每次正式实验必须记录：

- Git commit；
- 配置文件；
- 数据合同 hash；
- 外部特征 manifest hash；
- 随机种子；
- 训练样本范围；
- 最佳 epoch 和 early-stopping monitor；
- 四个验证场景的指标；
- chemical/genome correct、shuffle、zero 状态；
- 是否启用批次、相似药物迁移和蛋白先验。

## 15. 提交接口

提交前必须满足：

- 第一列为 `sample_ID`；
- 行与测试 metadata 一一对应；
- 蛋白列名称、数量和顺序与冻结 feature contract 完全一致；
- 全部为有限 log2 数值；
- 无 NA、无 inf；
- 预测生成过程未读取测试蛋白真值。

