# GOAI V2 化学外部特征可复核报告

## 1. 任务与边界

本报告覆盖“比赛化合物名称 - PubChem 结构 - RDKit 特征”模块。构建只读取两份 metadata 的四个字段：`perturbation_no_concentration`、`pert_id`、`chemical_role`、`split_final`。脚本不接受、读取或搜索 proteome 路径，也没有读取任何训练、验证或测试蛋白标签。

未修改模型、训练代码、`project_v2` 合同、Water/DMSO matched-control 映射或其他任务模块。未执行 Git 操作。

## 2. 输入与实体盘点

- 训练/验证 metadata：8,958 行。
- 测试 metadata：4,454 行。
- 合计：13,412 行，57 个唯一 `perturbation_no_concentration`。
- 固定首次出现优先级：`train`、`val_chem_only`、`val_strain_only`、`val_both`、`val_time`、`test_chem_only`、`test_strain_only`、`test_both`、`test_time`。
- `pert_id` 仅作为集合保留，未用作身份键。真实数据存在同一 `pert_id` 对应不同名称的情况。

每个实体保存全部 `pert_id`、`chemical_role`、`seen_splits`、`seen_in_train`、`seen_in_validation`、`seen_in_test` 和 metadata 行数。

## 3. PubChem 查询与人工复核

PubChem 使用 PUG-REST 名称候选、CID 属性和 CID 同义词接口。请求间隔至少 0.25 秒（不超过 4 次/秒），默认超时 30 秒、最多重试 3 次并指数退避。每个原始小型响应保存为 JSON 信封，记录 URL、HTTP 状态、日期、原始 HTTP 响应体 SHA-256 和缓存文件 SHA-256。

PubChem 当前属性字段使用：

- `ConnectivitySMILES`：连接性 SMILES；
- `SMILES`：含立体/同位素信息的 SMILES；
- `InChIKey`、`MolecularFormula`、`IUPACName`、`Title`。

旧称 `CanonicalSMILES` / `IsomericSMILES` 仅作为兼容回退；`compound_mapping.csv` 逐行记录实际使用字段名。

Main 定点修正前状态为 `confirmed=49`、`proxy=1`、`special_control=3`、`unresolved=4`。本次复核后最终状态：

| 状态 | 数量 | 说明 |
|---|---:|---|
| confirmed | 52 | 其中 Cisplatin、NaCl 身份确认但当前结构特征无效 |
| proxy | 1 | `Oligomycin` 未给组分，显式使用 Oligomycin A 代理 |
| special_control | 3 | Water、DMSO、Quality Control，模型化学特征保持关闭 |
| unresolved | 1 | Tunicamycin |

三项身份定点修正：

- U-73122：MeSH 将 U-73122 与 CAS `112648-68-7` 及 17beta 化学名绑定；Tocris 同时给出该 CAS、CID `104794`、立体 SMILES、式和 InChIKey；PubChem CID `104794` 一致。最终 `confirmed/high`，InChIKey `LUFAORPFSVMJIW-ZRJUGLEFSA-N`。
- Cisplatin：官方 PubChem Cisplatin 记录为 CID `5460033`。身份 `confirmed/high`；完整 `N.N.Cl[Pt]Cl` 保留，不拆分、不去配位组分。因当前普通有机分子 Morgan/RDKit 特征体系不适用，`structure_valid=false`、零特征、mask=false。
- NaCl：官方 PubChem Sodium Chloride 记录为 CID `5234`，完整离子结构 `[Na+].[Cl-]`。身份 `confirmed/high`；不选择单个离子、不去盐。当前特征体系不适用，`structure_valid=false`、零特征、mask=false。

Tunicamycin 仍为 unresolved：家族/同系物混合名称，metadata 未给具体组分。

`manual_overrides.csv` 是人工决定的最高优先级输入；自动查询不得覆盖它。`manual_review.csv` 保存候选摘要、最终选择、问题类型、理由、证据 URL 和复核日期。

## 4. 结构标准化

原始 PubChem SMILES、原始完整 InChIKey、标准化母体 SMILES、母体 InChIKey和每一步操作均保留。默认步骤为：

1. `Chem.MolFromSmiles(sanitize=True)`；
2. `rdMolStandardize.Cleanup(default parameters)`；
3. 仅对人工 override 明确标记 `fragment_parent` 的盐/水合物/溶剂化物执行 `rdMolStandardize.FragmentParent(default parameters)`；
4. `Chem.SanitizeMol` 和 `Chem.GetSymmSSSR`。

没有静默去盐。无法合理确定母体的多组分实体保持 unresolved。

## 5. 特征与 scaler

- RDKit：2026.03.5。
- Morgan：radius=2，2,048 bits，`useChirality=true`，二进制、不标准化。
- RDKit descriptors：217 个，顺序来自该版本 `Descriptors._descList`。
- 拼接顺序：Morgan 2,048 维 + 标准化 descriptors 217 维 = 2,265 维。
- 连续 descriptor scaler：`ddof=0`，仅用 `seen_in_train=true`、`mapping_status=confirmed`、结构有效、非特殊对照的 33 个普通训练化合物拟合。
- 44 个零方差 descriptor 的 scale 固定为 1；完整名称列表在 `feature_schema.json`。
- unresolved、special_control，以及身份 confirmed 但 `structure_valid=false` 的 Cisplatin/NaCl：数值特征全 0，`feature_valid_mask=false`。

`feature_schema.json` 冻结并哈希 descriptor 顺序、实体顺序和 scaler 内容。

## 6. 相似度与反证接口

Tanimoto 使用同一套标准化母体 Morgan 指纹，矩阵为“34 个训练普通有效药物 × 51 个全部普通有效 confirmed/proxy 药物”，形状 34 x 51。行列顺序单独保存。特殊对照、unresolved 和 `structure_valid=false` 的实体不进入相似度矩阵。

脚本提供 `apply_feature_variant`，支持 `correct`、`shuffle`、`zero`。`shuffle` 必须同时传入 `mapping_status`、`structure_valid`、`special_control_type`，仅在 confirmed/proxy、结构有效、非特殊对照实体之间重排，并返回完整的“目标行 -> 来源行” permutation。Water、DMSO、Quality Control、unresolved 和所有结构无效实体保持原位。shuffle 后有效药物特征多重集不变，且不会收到 fallback 零特征；默认种子 20260813。此任务未训练模型，因此没有运行模型消融指标。

## 6.1 标准化描述符极端值诊断

本轮未改变冻结特征定义，只做诊断。对所有结构有效实体应用 train-only 冻结 scaler 后，绝对 z 值分布为：中位数 0.376、P90 1.312、P95 1.947、P99 3.974、P99.9 7.831、最大值 27.270；`|z|>5` 共 58 个单元，`|z|>10` 共 7 个单元。

最大极端项包括：H2O2 的 `BCUT2D_MWLOW` 27.27 和 `MinAbsEStateIndex` 18.23；Neomycin B 的 `fr_NH2` 24.89 和 `VSA_EState4` 14.74；G418 的 `fr_NH2` 12.32；Pentamidine isethionate 与 FCCP 的 `SMR_VSA2` 分别为 10.82 和 10.52。完整 top-20 位于 `feature_schema.json`。

建议 Main 后续以严格 train-only 方式独立比较：（1）在训练化合物上估计 clipping 界限后冻结应用；（2）使用训练化合物中位数/IQR 的稳健标准化。本轮未执行 clipping，也未替换当前 z-score。

## 7. 可复现运行

联网候选发现（首次或补缓存）：

```powershell
D:\虚拟细胞\.venv\Scripts\python.exe D:\虚拟细胞\scripts\build_chemical_features.py --metadata-train-val "D:\虚拟细胞\WAYB_WAYC\WAYB_WAYC_metadata_train_val(1).csv" --metadata-test "D:\虚拟细胞\WAYB_WAYC\WAYB_WAYC_metadata_test(1).csv" --output-dir "D:\虚拟细胞\external_data\chemistry" --overrides "D:\虚拟细胞\external_data\chemistry\manual_overrides.csv" --discovery-only
```

人工复核后离线重建：

```powershell
D:\虚拟细胞\.venv\Scripts\python.exe D:\虚拟细胞\scripts\build_chemical_features.py --metadata-train-val "D:\虚拟细胞\WAYB_WAYC\WAYB_WAYC_metadata_train_val(1).csv" --metadata-test "D:\虚拟细胞\WAYB_WAYC\WAYB_WAYC_metadata_test(1).csv" --output-dir "D:\虚拟细胞\external_data\chemistry" --overrides "D:\虚拟细胞\external_data\chemistry\manual_overrides.csv" --offline
```

测试：

```powershell
D:\虚拟细胞\.venv\Scripts\python.exe -m pytest D:\虚拟细胞\tests\test_chemical_features.py -q
```

最终实际结果：`14 passed`。测试中的离线重建会复制缓存到 pytest 临时目录，重新生成全部核心工件，并逐字节比较 CSV/schema/index、逐数组比较 NPZ 数值；另逐项校验 manifest 记录的输入、派生工件和全部 PubChem 缓存文件 SHA-256，并验证受限 shuffle 的资格、permutation、fallback 隔离和特征总体分布不变。

## 8. 外部资源、版本与许可

- PubChem PUG-REST：动态服务，访问日期 2026-08-13；逐请求 URL、缓存与 SHA-256 在 `source_manifest.json`。PubChem/NCBI 许可说明链接亦记录在 manifest。
- RDKit 2026.03.5，BSD-3-Clause。
- `references/20260812Approach.pdf`：本地参考，完整 6 页已读取并渲染检查。
- `references/OfficialRules.pdf`：本地规划版规则，完整 36 页已读取并渲染检查；规则声明最终以组委会正式发布为准。
- Datawhale 优化教程：动态网页在本次自动访问中不可读取；仅作为上下文引用，未从该页取数据或决定实体映射。此项属于外部参考访问限制，不影响从 metadata + PubChem 缓存离线重建。

## 9. 失败记录与限制

1. 项目 `.venv` 初始缺少 RDKit、pytest、requests，已仅安装到项目虚拟环境。
2. 首次联网请求被沙箱网络策略拦截；获准后按断点缓存继续。
3. 首版缓存 hash 错将整个缓存信封作为“原始响应”hash；该草稿曾隔离用于诊断，最终交付前已清理。最终缓存同时记录原始 HTTP 响应体和缓存文件 hash。
4. 首次离线特征构建触发 RDKit `RingInfo not initialized`；修复为标准化后显式 sanitize/ring initialization，并由最终测试覆盖。
5. 本模块没有也不应决定 Water/DMSO matched-control 规则。

## 10. 结论与待 Main 决定项

化学工件已满足本模块文件接口。函数级接口发生一项有意收紧：`apply_feature_variant(..., mode="shuffle")` 现在要求显式传入三项资格元数据，并返回受限 permutation；NPZ 主特征维度和文件合同不变。建议 Main 接入时按新签名调用。Tunicamycin 仍需供应商/配方证据才能解除 unresolved；Oligomycin A 保持 `proxy/medium`。
