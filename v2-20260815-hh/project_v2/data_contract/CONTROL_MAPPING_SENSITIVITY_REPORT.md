# Water/DMSO 映射敏感性测试报告

生成日期：2026-08-13

## 结论

本报告是现有数据合同之上的独立、仅 train 的敏感性分析，不推翻原审计。官方 Water/DMSO 映射继续为 `BLOCKED_PENDING_ORGANIZER_CONFIRMATION`。奇偶规则只标记为 `inferred_mapping`，仅可用于实验性 train FC loss；合并均值规则只作诊断对照。不得依据验证或测试分数在两者间选择。

## 已确认事实

- 所有蛋白计算仅使用 `split_final=train` 的 5920 个样本和已冻结的 4422 个蛋白。
- 未打开 test proteome；没有将 validation protein cells 载入计算数据框或统计量。
- exact context 键：`data_source`, `Strains`, `Medium`, `Temperature`, `pert_time`, `pert_time_unit`, `instrument`, `Yeast_cell_plate`。
- `#48` 在 train 中有 91 行，全部为 Quality Control：True。

## 冻结的实验规则

- `pert_id #1–#47`：奇数 → Water，偶数 → DMSO。
- `#48` → Quality Control，排除于 treatment/control FC 配对。
- 该映射是 `inferred_mapping`，不是主办方确认规则，不可用于官方 FC 口径。
- 对照规则：同一 exact context 下所有 Water/DMSO control 按蛋白、仅在有效观测上合并均值。

## 匹配覆盖率

| 规则 | 可配 treatment | 成功匹配 | 失败 | 覆盖率 | 失败原因 |
|---|---:|---:|---:|---:|---|
| inferred parity | 5078 | 4785 | 293 | 94.230012% | `{"no_exact_expected_control": 293}` |
| pooled context | 5078 | 5066 | 12 | 99.763686% | `{"no_exact_water_or_dmso_control": 12}` |

pooled 成功匹配样本的 control 构成：`{"both_water_and_dmso": 4503, "water_only": 170, "dmso_only": 393}`。奇偶规则失败按预期溶剂分解：`{"DMSO": 90, "Water": 203}`。

原审计中的候选覆盖率诊断允许从全部 split 的元数据中寻找 control；本次按新增要求将 treatment 与 observed control label 池都严格限制为 train。因此两组覆盖数字不可直接互换，这一差异不修改原审计事实或官方 BLOCKED 状态。

## FC 敏感性结果

为避免覆盖差异造成混杂，以下核心比较仅使用两条规则都成功匹配的 4785 个 treatment，并使用共同有效 mask（17319160 个样本×蛋白位置）。

| 指标 | 结果 |
|---|---:|
| FC Pearson（全局展平、共同 mask） | 0.928923747 |
| 方向一致率 | 87.028527942% |
| FC 平均绝对差 | 0.101402164 |
| 两规则 FC 同为精确 0 的位置 | 9 |

方向一致定义为共同 mask 上 `sign(FC_parity) == sign(FC_pooled)`；精确零仅在符号相等时计为一致。

### 高效应蛋白（严格 `abs(FC) > 1`）

| 统计范围 | 规则 | 样本×蛋白位置数 | 去重蛋白数 | 每 treatment 中位数 |
|---|---|---:|---:|---:|
| 各自全部成功匹配样本 | inferred parity | 578269 | 4420 | 75.000 |
| 各自全部成功匹配样本 | pooled context | 538404 | 4421 | 61.000 |
| 共同匹配且共同有效 | inferred parity | 578269 | 4420 | 75.000 |
| 共同匹配且共同有效 | pooled context | 479256 | 4421 | 54.000 |

“去重蛋白数”表示至少在一个纳入样本中达到阈值的 frozen protein 数；它不等同于高效应事件数，所以同时给出样本×蛋白位置数。

## 防泄漏与使用限制

- 敏感性规则在 train 元数据上冻结，未使用 validation/test 分数选择。
- 混合 train/validation proteome CSV 先只读取 `sample_ID`；蛋白读取阶段用 `skiprows` 在解析蛋白单元格之前排除全部非 train 行。
- 因源文件物理上混合 train/validation，顺序扫描文件字节不可避免；但 validation 蛋白单元格不被 tokenization、物化或用于任何统计。
- 脚本不接受或打开 test proteome 路径。
- `inferred_parity` 仅供实验性 FC loss；`pooled_context` 仅为敏感性比较；两者都不得替代官方 BLOCKED 映射。
- 本次未下载或引入任何外部资源，因此无新增 URL、许可或外部资源校验记录。

## 尚待主办方确认

- 权威 `pert_id → Water/DMSO` 映射。
- 是否存在 exact context、溶剂匹配之外的样本级额外 QC。
- 多个匹配 control 的官方聚合规则。
