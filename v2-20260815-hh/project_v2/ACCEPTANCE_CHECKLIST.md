# GOAI 虚拟细胞 V2 验收清单

## 1. 验收原则

每个模块按以下顺序验收：合规性 → 接口一致性 → 正确性 → 科学合理性 → 消融证据 → 稳定性 → 性能。前一层未通过，不因单次验证分数较高而跳过。

验收状态仅使用：

- `PASS`：有可复核证据且满足要求；
- `FAIL`：已确认不满足；
- `BLOCKED`：缺少外部证据或上游接口，不能判断；
- `NOT RUN`：尚未执行。

## 2. 所有线程通用验收

- [ ] 未修改或直接推送 `main`。
- [ ] 在指定分支和目录工作。
- [ ] 提供 branch、commit 和文件清单。
- [ ] 提供实际运行命令、环境和依赖版本。
- [ ] 提供测试结果，而非只写“已运行”。
- [ ] 报告失败实验、未完成项和假设。
- [ ] 未上传比赛原始数据。
- [ ] 外部资源有 URL、版本、日期、许可和 SHA-256。
- [ ] 验证/测试蛋白标签未用于训练、特征、过滤、标准化、降维或统计先验。
- [ ] 接口变更经过 Main 批准。

## 3. 数据合同验收

### 3.1 对齐与划分

- [ ] metadata/proteome 按 `sample_ID` 对齐。
- [ ] `sample_ID` 完整、唯一，缺失和重复均有检查。
- [ ] train、val_chem_only、val_strain_only、val_both、val_time 真实样本数有报告。
- [ ] test 各场景样本数有报告。
- [ ] 所有训练统计函数显式限制为 train IDs。

### 3.2 蛋白列表

- [ ] 缺失率只由 train 标签计算。
- [ ] 阈值的数学条件和边界值有自动测试。
- [ ] raw intensity 的 0、负值、NA、inf 处理有明确规则。
- [ ] Quality Control 是否纳入 train 统计有证据和说明。
- [ ] 4,422 与 4,232 差异有逐步证据链。
- [ ] 若仍无法解释，报告明确列出缺失证据，且未强行匹配教程数字。
- [ ] feature contract 包含蛋白名称、顺序、数量和生成 hash。

### 3.3 Control 与 FC

- [ ] Water/DMSO 映射明确。
- [ ] exact matching 字段冻结。
- [ ] 处理样本无匹配 control 的原因有分类统计。
- [ ] `fc_true` 仅在 treatment/control 共同有效位置计算。
- [ ] 官方 `fc_pred` 与模型内部 `delta_response` 清楚区分。
- [ ] Pearson 使用严格共同有效样本/位置。
- [ ] R² 和其他指标均为 mask-aware。
- [ ] 验证场景指标与可信基线可解释地一致，或差异有归因。

## 4. 化学外部特征验收

- [ ] 从真实 metadata 自动提取全部唯一化合物。
- [ ] 每个化合物标注首次出现的 split，但未读取其蛋白标签。
- [ ] 原始名称、标准名称、CID、SMILES、InChIKey 均有保存。
- [ ] 盐型、立体异构体、混合物和多候选匹配有人工复核记录。
- [ ] Water、DMSO、Quality Control 等特殊处理未被错误编码为普通化合物。
- [ ] 未静默去盐；原始结构和标准化母体结构可追溯。
- [ ] 所有确认 SMILES 可被 RDKit 解析。
- [ ] Morgan 位数、半径和 chirality 设置已冻结。
- [ ] descriptor 名称和顺序已冻结。
- [ ] 特征无未解释的 NA/inf。
- [ ] descriptor 标准化只在训练化合物上拟合。
- [ ] unresolved 化合物采用显式 fallback。
- [ ] Tanimoto 相似度矩阵可重复生成。
- [ ] correct/shuffle/zero 接口可用且随机种子固定。

## 5. 菌株基因组验收

- [ ] BAH、BAI、CEK、CGD、CRD、DHY210 全部有映射状态。
- [ ] 每个精确/支持映射都有公开证据，不仅依赖缩写相似。
- [ ] BAI、CRD 的证据单独审核。
- [ ] DHY210→S288C 明确标为 proxy，而非 exact。
- [ ] unresolved 菌株采用显式 unknown/low-confidence 回退。
- [ ] FASTA/VCF 版本、来源和校验值记录完整。
- [ ] 简单变异和通路聚合特征可重复生成。
- [ ] PCA/标准化只在训练菌株上拟合。
- [ ] 验证/测试菌株仅使用冻结转换。
- [ ] correct/shuffle/zero 接口可用。
- [ ] 在简单特征尚未通过前，没有擅自引入 Genome LM。

## 6. V2 模型合同验收

### 6.1 分支输入隔离

- [ ] 基础状态分支看不到 chemical 和 batch。
- [ ] 响应分支看不到 batch。
- [ ] 批次分支看不到 chemical、genome、strain、medium、temperature、time。
- [ ] 输入隔离由测试验证，不仅靠文档声明。

### 6.2 输出恒等式

- [ ] 返回 `y_pred`、`y_baseline`、`delta_response`、`delta_batch`。
- [ ] 自动测试验证 `y_pred = y_baseline + delta_response + delta_batch`。
- [ ] control 的 `delta_response=0` 或约束机制有测试。
- [ ] 输出维度从 feature contract 读取，没有硬编码。
- [ ] 输出为 log2 尺度。

### 6.3 相似药物迁移

- [ ] FC anchor 只由 train 标签计算。
- [ ] 验证/测试仅通过结构相似度查询 train anchor。
- [ ] 邻居数量、相似度阈值和低相似度回退有配置。
- [ ] 门控权重有范围测试和日志。
- [ ] 无相似药物分支的消融已运行。

### 6.4 损失和训练

- [ ] mask-aware Huber/MSE 单独可运行。
- [ ] FC loss 使用冻结后的定义与 mask。
- [ ] 高效应阈值只从 train 估计。
- [ ] 每个损失分项单独记录。
- [ ] 总 loss 严格等于各项加权和。
- [ ] 未启用损失权重为零且日志可见。
- [ ] 早停真实触发并恢复最佳 epoch。
- [ ] checkpoint 包含配置、feature contract hash 和特征 manifest hash。
- [ ] 同一随机种子可复现；多个种子用于稳定性判断。

## 7. 模型实验验收

### 7.1 必须存在的对照

- [ ] Protein Mean。
- [ ] Matched Control。
- [ ] 旧 ConditionMLP。
- [ ] Morgan + descriptors MLP。
- [ ] 相似药物 FC 模型。
- [ ] 对照锚定 V2 模型。

### 7.2 必做消融

- [ ] chemical correct vs shuffle vs zero。
- [ ] genome correct vs shuffle vs zero。
- [ ] 有/无相似药物迁移。
- [ ] 有/无批次分支。
- [ ] 正确批次 vs 批次字段打乱。
- [ ] Huber only vs Huber + FC。
- [ ] 有/无高效应蛋白加权。
- [ ] 加入蛋白先验时，有/无该先验。

### 7.3 分场景报告

- [ ] val_chem_only。
- [ ] val_strain_only。
- [ ] val_both。
- [ ] val_time。
- [ ] 留出化学骨架内部 CV。
- [ ] 留出菌株内部 CV。
- [ ] 双重留出内部 CV。
- [ ] 留出 plate/instrument 的批次稳定性验证。

### 7.4 结论纪律

- [ ] correct embedding 稳定优于 shuffle/zero 后才宣称外部表征有效。
- [ ] 单个种子提升不作为稳定结论。
- [ ] 只改善 Global R²、不改善 FC/残差时不宣称学到扰动机制。
- [ ] 批次分支仅在公开切分有效、批次留出崩溃时不作为主模型贡献。
- [ ] 失败结果被保留并归因。

## 8. 蛋白先验验收

- [ ] GO/KEGG/STRING 来源、版本和许可记录完整。
- [ ] 比赛蛋白 ID 到外部数据库 ID 的映射有覆盖率和冲突报告。
- [ ] 图中无非预期 self-loop。
- [ ] 相关性图使用严格成对 mask-aware 计算。
- [ ] 若使用数据驱动图，只由 train 标签构建。
- [ ] 优先在 FC/残差或低秩 basis 上建图，而非原始丰度简单平滑。
- [ ] 图/通路模块有独立消融。
- [ ] 未改善目标场景时可完全关闭并回退。

## 9. 最终提交验收

- [ ] `prediction.csv` 行数与测试 metadata 一致。
- [ ] 第一列为 `sample_ID`，完整且唯一。
- [ ] 行顺序可通过 sample_ID 校验。
- [ ] 蛋白名称、数量和顺序与 feature contract 一致。
- [ ] 预测值为 log2。
- [ ] 无 NA。
- [ ] 无 inf。
- [ ] 生成预测的流程未读取测试蛋白真值。
- [ ] 外部数据来源和许可证披露完整。
- [ ] 固定随机种子、环境、运行说明和模型配置齐全。
- [ ] 从干净环境可复现最终预测。

## 10. 执行线程标准交付格式

```text
任务：
工作分支：
commit：
新增/修改文件：
实际使用的数据：
外部资源与版本：
运行命令：
测试结果：
核心实验结果：
与基线比较：
失败或未完成项：
是否涉及接口变更：
需要 Main 决定的问题：
推荐下一步：
```

## 11. Main 最终准入判断

一个模块只有在以下条件全部满足时，才允许进入主模型：

1. 无标签泄漏和外部数据合规问题；
2. 接口与本合同一致；
3. 自动测试覆盖核心不变量；
4. 科学映射和假设可解释；
5. 有 correct/shuffle/zero 或等价反证实验；
6. 多种子和目标 OOD 场景稳定；
7. 提升不能主要来自批次捷径；
8. 有明确失败回退。

