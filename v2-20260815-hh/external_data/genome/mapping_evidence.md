# 比赛菌株—公开酵母基因组映射证据

## 判定口径

- `exact` 仅用于公开来源明确给出比赛名称与外部样本的一对一关系。本轮没有菌株达到这一更严格门槛。
- `supported` 要求至少两条独立可信证据闭合身份链；Peter et al. 2018 Supplementary Table S1 是主证据，另一数据库、论文或学术复分析为独立复核。
- `proxy` 不是身份等价。本轮仅 DHY210 使用 S288C 代理。
- 所有三字母代码都来自真实 metadata 自动盘点；没有依靠缩写相似性做首条命中。

主资源为 Peter et al. 2018 的 1,011 株酵母研究及其 Supplementary Table S1。该表中的 `Standardized name` 与公开矩阵行名一致；原始补充表 SHA-256 为 `b11de13b50bcf91bb2a40cdbd0f2f35372bdef6fc9a018d7538cf5e5eea7f273`。由于尚未找到比赛数据提供方明确声明“比赛三字母代码就是 Peter 表标准名”的一对一文件，五个直接公共菌株保守标为 `supported`，不标为 `exact`。

## BAH

- 结论：`supported/high`；`external_isolate_id=SX3`；`external_matrix_id=BAH`。
- 主证据：Peter Table S1 的 `Isolate name=SX3, Standardized name=BAH`。
- 独立证据：[NCBI BioProject PRJNA396809](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA396809)列出 SX3 的公开组装；[Rinta-Harri et al. 2026](https://doi.org/10.1111/1751-7915.70337)的菌株表明确并列 SX3 与 BAH。
- 候选与拒绝：唯一接受候选为 SX3；未找到需要择一的同名候选。任何只因 `BAH` 缩写相似而出现的候选均被规则性拒绝。

## BAI 单独审计

- 泛化角色：BAI 不在 train；由 metadata 的冻结 split 顺序判定 `first_seen_split=val_strain_only`，因此只应用训练菌株拟合的冻结变换。
- 结论：`supported/high`；`external_isolate_id=BJ6`；`external_matrix_id=BAI`。
- 主证据：Peter Table S1 的 `Isolate name=BJ6, Standardized name=BAI`。
- 独立证据：[NCBI BioProject PRJNA396809](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA396809)列出 BJ6；[ScRAPdb](https://evomicslab.org/db/ScRAPdb/strains/)独立列出 BJ6 与 BAI/BAI_1a 别名。
- 候选与拒绝：唯一接受候选为 BJ6；没有把缩写相似但无身份链的记录纳入候选。
- 风险：比赛数据提供方的一对一身份声明仍未获得，因此未提升为 `exact`。

## CEK

- 结论：`supported/medium`；`external_isolate_id=JCM_2985-4B`；`external_matrix_id=CEK`。
- 主证据：Peter Table S1 明确配对 CEK 与 JCM_2985-4B。
- 独立证据：[University of Washington 复分析附表](https://digital.lib.washington.edu/researchworks/bitstreams/1fe8f5c6-1b14-4bd1-abb6-14f74c0ea9be/download)重复给出 `CEK / JCM_2985-4B / ERR1309167`。
- 候选与拒绝：JCM_2985-4B 是接受候选，ERR1309167 是序列读段辅助 ID；没有把后者误写成 isolate 名。

## CGD

- 结论：`supported/medium`；`external_isolate_id=UCD_09-448`；`external_matrix_id=CGD`。
- 主证据：Peter Table S1 明确配对 CGD 与 UCD_09-448。
- 独立证据：[NCSU 学术复核表](https://repository.lib.ncsu.edu/server/api/core/bitstreams/5b4e10ff-fd0e-4e8e-9261-e2ca87dadb03/content)及[UPV 学术表](https://riunet.upv.es/bitstreams/f9c19cb9-cf10-4018-a43e-6cbab176301d/download)重复给出该配对。
- 候选与拒绝：只接受 UCD_09-448；没有直接比赛提供方声明，故保持 medium。

## CRD 单独审计

- 泛化角色：CRD 只出现在 test；`first_seen_split=test_strain_only`，完全不参与缺失值、scaler、方差筛选或 PCA 拟合。
- 结论：`supported/medium`；`external_isolate_id=FIMA_3`；`external_matrix_id=CRD`。
- 主证据：Peter Table S1 的 `Isolate name=FIMA_3, Standardized name=CRD`，来源注明意大利 wine isolate。
- 独立证据：[van Leeuwen et al. 2020](https://doi.org/10.15252/msb.202110160)独立描述 FIMA_3 为 Peter et al. 的意大利酒庄菌株。
- 候选与拒绝：唯一接受候选为 FIMA_3。独立论文未重复三字母代码，因此没有夸大为 `exact/high`。

## DHY210

- 结论：`proxy/low`；`external_isolate_id=S288C`，`external_assembly_id=GCF_000146045.2`，参考坐标 R64-1-1。
- 证据：[NCBI S288C R64](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000146045.2)给出参考装配身份；本地 `references/20260812Approach.pdf` 允许将 S288C 作为 DHY210 的工程代理。
- 明确拒绝：拒绝 `exact` 或“等价菌株”表述。没有公开一对一证据证明 DHY210=S288C。
- 信息损失：代理会丢失 DHY210 特有 SNP、indel、CNV、基因缺失、杂合与结构变异；所以 `proxy_flag=true`，并永久排除在所有拟合统计量之外。

## 资源与许可风险

Peter 论文和未另作说明的补充材料采用 [CC BY 4.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC6784862/)。1002 公共文件服务器没有在文件索引中单独声明原始矩阵复用许可，因此 manifest 如实记录“论文为 CC BY 4.0、原始服务器条款未单列”，不把两者静默等同。完整 URL、版本、日期、大小与 SHA-256 位于 `raw_resource_manifest.json`。
