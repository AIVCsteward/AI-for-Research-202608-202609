# GOAI WAYB/WAYC 数据与评测口径审计报告

审计日期：2026-08-13  
审计范围：`D:\虚拟细胞\WAYB_WAYC` 的四个真实比赛 CSV；项目章程、接口合同、验收清单；`20260812Approach.pdf`；`OfficialRules.pdf`。  
实现范围：仅 `project_v2/data_contract/`，未修改模型代码。  
标签边界：最终可复核脚本和 feature contract 只使用 `split_final=train` 的蛋白标签拟合统计量；对测试 proteome 只读取 `sample_ID` 列，且不做全文件哈希。初始人工探索曾错误地计算两项全 train_val 缺失率敏感性，并发生过一次测试 proteome 字节级哈希和 `nrows=3` 结构查看；这些结果未用于训练、特征、冻结筛选、control 推断或合同决策。过程偏差在第 8、10 节明确记录。

## 1. 结论与当前合同状态

| 项目 | 状态 | 冻结结论 |
|---|---|---|
| metadata/proteome 对齐 | PASS | 必须以 `sample_ID` 做集合检查和重排；当前两个文件对均完整、唯一、一一对应，源文件行序也恰好一致，但代码不得依赖行序。 |
| split 数量 | PASS | 已由真实 metadata 复算，见第 3 节。 |
| 蛋白 feature contract | PASS（当前真实文件） | 5,243 个原始蛋白中，以 5,920 个 train 样本、`invalid_rate < 0.80` 保留 4,422 个；顺序见 `feature_contract.json`。 |
| 4,422 / 4,232 差异 | BLOCKED（缺教程源文件/处理日志） | 已排除阈值方向、0/负值、inf、QC 纳入与误用全 train_val 等常见原因；4,232 需要约 69.7% 缺失阈值或未披露额外质控/不同文件版本，不能强制裁剪。 |
| exact context key | PASS | `data_source, Strains, Medium, Temperature, pert_time, pert_time_unit, instrument, Yeast_cell_plate` 全部严格相等。 |
| Water/DMSO 官方映射 | BLOCKED | 规划版规则要求逐处理选择 Water 或 DMSO，但发布 metadata 无 solvent 字段、参考材料无权威映射表；接口按 `pert_id` 接收显式映射，缺失即 fail-closed。 |
| FC 定义与 mask | PASS | `fc_true=y_treatment_observed-y_control_observed`；`fc_pred_official=y_treatment_predicted-y_control_observed`；只在 treatment/control/预测共同有限且双观测 mask 为真的位置计算。 |
| Pearson / R² | PASS（本审计实现） | 本目录实现严格 mask-aware，并有填充值不影响结果的测试。现有模型代码状态见第 7 节。 |
| 防标签泄漏 | PASS（最终实现）/ FAIL（初始探索过程） | 所有统计拟合接口显式要求 `fit_sample_ids` 并拒绝任一非 train ID；最终脚本对测试 proteome 只读 ID 列。初始探索曾对验证标签做两项不合规敏感性统计，并读取少量测试值/文件字节；均未流入合同生成。 |

在 Main 或主办方提供权威 Water/DMSO 映射前，`control_matching_spec.json` 的正式状态为 `blocked_pending_organizer_confirmation`。训练线程可以直接调用候选匹配、FC 和指标函数，但不得将奇偶 `pert_id` 候选规律标为“官方 FC”。

## 2. 证据类别

### 2.1 已确认事实

- 四个真实 CSV 的字段、行数、`sample_ID` 集合与蛋白列均由本次脚本实际读取。
- train 缺失过滤只使用 5,920 个 `split_final=train` 样本。
- 当前 train raw intensity 中有 8,590,321 个 NA；0、负值、`+inf`、`-inf` 均为 0 个。
- 当前文件中没有恰好 80% 无效的蛋白，因此 `<0.80` 与 `<=0.80` 在这批数据上都得到 4,422；合同仍冻结为教程文字对应的严格 `<0.80`。
- Quality Control 是独立处理名，不是 Water/DMSO，也不构造 treatment FC。train 中有 91 个 QC 样本。
- 当前 metadata 的 `pert_id` 不是全局唯一化合物 ID：例如不同数据布局下同一编号可对应不同名称；Water 为 `#1`，DMSO 为 `#2` 或 `#12`，QC 为 `#48`。
- 多数 exact context 同时存在 Water 和 DMSO，因此把两者无差别放入一个 lookup 无法确定官方 control。

### 2.2 教程或规划版规则中的数值/口径

以下内容来自参考 PDF，不等同于本次真实文件计算：

- `20260812Approach.pdf`：train 5,920；5,243 个原始蛋白按 80% 缺失阈值保留 4,232、删除 1,011。
- 同一教程：完整验证样本数为 1,065 / 1,547 / 269 / 157；用于 matched-control 基线的子集为 1,015 / 1,293 / 266 / 128。
- `OfficialRules.pdf` 是“2026 年 7 月规划版本”，规定 `Δ_pred=ŷ_treat-y_control`、`Δ_true=y_treat-y_control`，并要求 control 匹配来源、菌株、培养基、温度、时间、仪器和板号，再按官方规则选 DMSO 或 Water。
- 规划版明确写“所有参照、归一化统计与对照匹配规则须仅用训练数据冻结”，但未在文件中发布逐 `pert_id` 的 Water/DMSO 表。

### 2.3 当前真实文件计算结果

本报告第 3–6 节及 `audit_results.json` 中的数值均来自当前四个真实 CSV。它们只对下列校验值对应的文件有效。

### 2.4 尚待主办方或 Main 确认

1. 教程生成 4,232 所用 proteome 文件的 SHA-256、发布时间、下载批次及蛋白列清单。
2. 教程除 `missing_rate < 0.80` 外是否做过 peptide/QC/污染物/低信号/样本级额外质控。
3. 权威 `pert_id -> Water/DMSO` 映射；同一化合物在不同 `pert_id`（如 Brefeldin A、MMS）下是否允许不同溶剂。
4. 教程 matched-control 子集在 strict context + solvent 之外是否排除了缺板孔、特定 control replicate、样本级 QC 或需要全蛋白可比性。
5. 多个 exact control replicate 的正式聚合口径；本合同当前采用逐蛋白、只在观测 control 上取均值。

## 3. 文件身份、对齐与划分

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `WAYB_WAYC_metadata_train_val(1).csv` | 904,646 | `9414f22d71e925a3b85544b49fde252613c87808d34738a84785003adb8131ef` |
| `WAYB_WAYC_proteome_raw_train_val.csv` | 289,769,736 | `a15d9a40a6ad4e8e84a4ce4ed08644fce78780d31ace5561928517c4a5fa7ccb` |
| `WAYB_WAYC_metadata_test(1).csv` | 461,272 | `42f2df9ea79f28da8344e96b5181edacc215744a858d1a4eaa729c2e1cc69d31` |
| `WAYB_WAYC_proteome_raw_test.csv` | 143,352,203 | 未计算；为避免读取测试真值字节，只用 `usecols=['sample_ID']` 解析 ID。 |

metadata 与 proteome 的 `sample_ID` 均无缺失、无重复、无单侧 ID。当前源行顺序相同；独立脚本仍显式按 ID 重排并验证一对一。

| split | 实际样本数 |
|---|---:|
| train | 5,920 |
| val_chem_only | 1,065 |
| val_strain_only | 1,547 |
| val_both | 269 |
| val_time | 157 |
| test_chem_only | 1,640 |
| test_strain_only | 1,534 |
| test_both | 1,129 |
| test_time | 151 |

train_val 合计 8,958；test 合计 4,454。

## 4. 蛋白 feature contract 与 4,422 / 4,232 证据链

### 4.1 冻结算法

对每个原始蛋白列 `p`：

```text
valid(i,p) = isfinite(raw(i,p)) AND raw(i,p) > 0
invalid_rate(p) = mean_i[NOT valid(i,p)],  i 仅属于 split_final=train
keep(p) = invalid_rate(p) < 0.80
```

仅对 `valid=True` 的 raw intensity 做 `log2`；无效位置保持 NA/false mask。当前真实 train 文件没有有限非正值或 inf，所以本次 `invalid_rate` 数值等于 NA rate，但合同不能把 NA-only 行为推广到未来文件。

结果：5,243 → 4,422，删除 821。蛋白顺序 SHA-256：`f112e7fffae4d3158ec9985a6c9cfa25e1a7dd40f847092561ea023bc1d8a45f`。完整顺序和生成 hash 在 `feature_contract.json`。

### 4.2 逐步排查

| 变体 | 样本范围 | 保留蛋白数 | 是否解释 4,232 |
|---|---:|---:|---|
| NA 缺失，`rate < 0.80` | train 5,920 | 4,422 | 否 |
| 非正/非有限视作无效，`rate < 0.80` | train 5,920 | 4,422 | 否；当前无 0/负/inf |
| `rate <= 0.80` | train 5,920 | 4,422 | 否；无恰好 80% 列 |
| 排除 train QC | 5,829 | 4,428 | 否，方向相反 |
| 仅 train QC | 91 | 4,212 | 否；且不合法 |
| train，阈值 0.75 | 5,920 | 4,323 | 否 |
| train，阈值 0.80 | 5,920 | 4,422 | 当前合同 |

按当前文件排序，若只调一个严格缺失阈值，恰好保留 4,232 所需阈值位于 `(0.6964527, 0.6974662]`，约为 69.7%，与教程声明的 80% 不符。因此：

- 已有证据排除阈值方向、raw 0/非正、非法值、QC 是否纳入和全 train_val 误用；
- 仍缺教程输入文件版本和额外 QC 处理日志；
- 当前 feature contract 冻结 4,422，不为匹配教程强删 190 个蛋白。

## 5. Matched control 规则

### 5.1 冻结的 exact context

处理与 control 必须同时满足以下字段完全相同：

```text
data_source
Strains
Medium
Temperature
pert_time
pert_time_unit
instrument
Yeast_cell_plate
```

`sample_ID` 是连接键；`protein_well` 不进入 context key；`Quality Control` 永远排除。control 候选只允许 `Water` 或 `DMSO`，且必须再通过显式 `pert_id -> solvent` 映射选择正确一种。不得换用另一个可用溶剂，不得放宽字段，不得 global fallback。

### 5.2 Water/DMSO 映射状态

当前正式映射为空，状态 BLOCKED。原因是：

- PDF 只描述“按官方规则匹配到 DMSO 或 Water”，未给映射表；
- metadata 无 `solvent`/`vehicle` 字段；
- 大多数 exact context 同时有 Water 与 DMSO；
- 本地 ZIP 和参考材料未发现额外映射文件。

从 plate 布局观察到候选规律：`#1..#47` 的奇数 `pert_id -> Water`、偶数 `pert_id -> DMSO`；`#48` 是 QC。该规律与 #1/#2/#12 control 布局一致，并把 `val_both` 精确复现为 266，但无法同时复现教程的 1,015 / 1,293 / 128，因此只记录为候选证据。

### 5.3 各场景候选映射覆盖与失败原因

下表使用上述**未确认的奇偶候选映射**，只用于审计覆盖，不得报告为官方 FC。失败原因互斥。

| split | 全部样本 | control | QC | treatment | 候选成功 | 缺 exact 预期溶剂 control |
|---|---:|---:|---:|---:|---:|---:|
| train | 5,920 | 751 | 91 | 5,078 | 4,951 | 127 |
| val_chem_only | 1,065 | 0 | 0 | 1,065 | 1,007 | 58 |
| val_strain_only | 1,547 | 190 | 24 | 1,333 | 1,313 | 20 |
| val_both | 269 | 0 | 0 | 269 | 266 | 3 |
| val_time | 157 | 15 | 3 | 139 | 135 | 4 |
| test_chem_only | 1,640 | 0 | 0 | 1,640 | 1,602 | 38 |
| test_strain_only | 1,534 | 188 | 24 | 1,322 | 1,322 | 0 |
| test_both | 1,129 | 0 | 0 | 1,129 | 1,125 | 4 |
| test_time | 151 | 14 | 2 | 135 | 131 | 4 |

正式 fail-closed 状态下，由于映射未确认，所有 treatment 的失败原因为 `unconfirmed_control_mapping`，官方 FC 成功数均为 0。这个结果是安全接口的预期行为，不是数据缺失。

教程的 matched-control 子集为 1,015 / 1,293 / 266 / 128；候选映射得到 1,007 / 1,313 / 266 / 135。差异表明教程还用了另一版映射、文件或额外样本级规则。

## 6. FC、官方预测 FC 与 mask

对一个 treatment 和一个或多个 exact matched control：

1. 多 control replicate 先按蛋白、仅在 control observed 且有限的位置取均值；没有任一有效 control 的蛋白保持无效。
2. `fc_mask = treatment_obs_mask AND control_obs_mask AND finite(y_treatment_observed) AND finite(y_treatment_predicted) AND finite(y_control_observed)`。
3. `fc_true = y_treatment_observed - y_control_observed`，只在 `fc_mask` 上有值。
4. `fc_pred_official = y_treatment_predicted - y_control_observed`，只在同一个 `fc_mask` 上有值。

模型内部 `delta_response` 不是自动等价的 `fc_pred_official`。若评估模型内部 delta，必须使用不同指标名并单独报告。

Pearson 与 R² 均先做 `mask AND finite(true) AND finite(pred)`，再压平或按预先声明的轴计算；不能先用 0/均值填充后把填充值纳入均值、方差、相关或残差平方和。常数真值或常数预测导致 Pearson/R² 未定义时返回 NA，不用 0 伪装有效分数。

## 7. 对现有实现的只读核对

本节只报告，不修改模型。

### 7.1 已满足或基本满足

- `baseline/evaluation.py` 的 Global R² 和逐蛋白 R² 使用传入 mask；`real_data_demo.metric_row` 也用 `mask & finite(prediction)`，R² 本身 mask-aware。
- `baseline/evaluation.py::compute_fold_change` 对 treatment/control 共同观测位置构造 `fc_mask`，多 control 按蛋白观测均值处理。
- `baseline/losses.py::fc_pearson_loss` 使用显式 `fc_mask` 且排除非有限值。

### 7.2 不满足冻结合同或尚未实现

- `baseline/evaluation.py` 文件头仍将“fold change Pearson 指标”列为升级方向，当前正式评估接口没有官方 FC PCC 分场景报告。
- 现有 `MATCH_KEYS` 缺少 `pert_time_unit`，与本合同八字段不一致。
- 现有 control lookup 把 Water/DMSO 合并并可能取第一个 control；没有 `pert_id -> solvent` 映射，不能称为官方 matched control。
- `matched_control_predict` 在找不到 exact control 时使用全局 control mean，并用 protein mean 填 control NA；这可作为预测基线 fallback，但不得用于 `fc_true`、官方 FC PCC 或 FC loss 标签。
- 现有 `build_matched_control_pairs` 会把同 context 的 Water/DMSO 全部混入 control replicate 均值；其双 mask 数值计算正确，但 control 语义未通过官方映射，因此 FC 标签口径不合格。
- `real_data_demo.matched_control_comparison` 从传入的整套 metadata/target 建 control lookup；若用于 val/test 诊断，会读取同一评估集的 control truth。这只能作为隔离的 truth diagnostic，不能用于训练统计、模型选择输入或正式可部署预测。旧输出中确实包含 test truth diagnostic；本次审计未使用这些测试分数建立任何合同或统计量。

因此，“当前 Pearson、R² 是否 mask-aware”的答案是：所审读的 R² 和 FC loss 数学实现基本 mask-aware；“当前官方 FC PCC 是否严格合规”的答案是：否，正式分场景指标尚未完整实现，且 Water/DMSO 语义映射未冻结。

## 8. 防泄漏设计与测试

- `select_proteins`、`fit_masked_mean` 等标签统计函数必须显式接收 `fit_sample_ids`。
- `require_train_fit_ids` 会验证 ID 存在、唯一，并逐个拒绝任何 `split_final != train` 的 ID。
- test proteome 仅通过 `pd.read_csv(..., usecols=['sample_ID'])` 做 ID 对齐；不读取、log2、过滤、聚合或哈希蛋白标签。
- `feature_contract.json` 明确记录 validation/test 标签未用于生成。
- 自动测试使用极端填充值证明 masked Pearson/R² 与被 mask 的值无关。

过程偏差：在最终脚本形成前的人工盘点中，曾（1）计算过全部 train_val 与“全部 train_val 排除 QC”两项不合规的缺失率敏感性；（2）使用 `pd.read_csv(..., nrows=3)` 查看四个 CSV 的表头/示例并计算四文件 SHA-256，因此测试 proteome 的前三行蛋白值和全文件字节被读取过一次。相关结果没有进入 4,422 列表、生成 hash、control 推断、特征合同或模型，正式报告也不把全 train_val 数字当作证据。后续代码不保存测试 proteome hash。为避免误导，本次交付把初始探索列为流程 FAIL；可复核的正式脚本和全部测试遵守 train-only / test-ID-only 访问。

测试环境：Python 3.12.13，pandas 3.0.1，NumPy 2.3.5；无新增依赖。

```text
python -m unittest -v test_data_contract.py
Ran 12 tests ... OK

python -m unittest -v test_real_data_contract.py
Ran 1 test ... OK
```

真实集成测试固定检查：九个 split 数量、5,243 → 4,422、验证/测试标签未用于合同生成、测试 proteome 仅 ID 访问。

## 9. 可复核命令与工件

```powershell
python project_v2/data_contract/audit_data_contract.py `
  --data-dir WAYB_WAYC `
  --output-dir project_v2/data_contract `
  --print-json

cd project_v2/data_contract/tests
python -m unittest -v test_data_contract.py
python -m unittest -v test_real_data_contract.py
```

交付工件：

- `DATA_CONTRACT_REPORT.md`：本报告。
- `feature_contract.json`：当前真实文件的 4,422 蛋白名称、顺序、生成与来源 hash。
- `control_matching_spec.json`：严格 context、FC、mask 和 fail-closed 映射接口。
- `audit_data_contract.py`：独立审计、训练 ID 守卫、匹配、FC 与 mask-aware 指标实现。
- `audit_results.json`：真实审计机器可读结果，非模型输入。
- `tests/`：快速单元测试和真实文件集成测试。

## 10. 失败项与推荐下一步

失败/未完成：

- 无法用已披露规则复现教程 4,232；缺教程源文件身份与额外质控证据。
- 无法从公开材料冻结官方 Water/DMSO 映射；候选奇偶规律不能完全复现教程 matched-control 数量。
- 初始人工探索对验证标签做过两项全 train_val 缺失率敏感性统计，并读取过测试 proteome 三行示例/全文件字节；未用于训练、冻结统计或特征，但违反严格标签边界。正式脚本已改为 train-only / test-ID-only，建议 Main 将本次过程偏差纳入验收记录。
- 因工作区缺少 `.git` 元数据，本次无法创建 `audit-data-contract` 分支或提交 commit；未初始化新仓库，以免改变协作边界。

推荐下一步：Main 向主办方索取并校验（1）教程 proteome SHA-256/蛋白列清单/预处理日志，（2）权威 `pert_id -> Water/DMSO` 映射，（3）教程 matched-control 额外排除与多 control 聚合规则。确认后只需填充 `control_matching_spec.json.confirmed_control_mapping`，更新状态并重跑集成测试；在此之前训练线程不得启用官方 FC loss/PCC。
