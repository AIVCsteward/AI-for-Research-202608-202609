# GOAI V2 菌株基因组映射与 V1 特征复核报告

任务：构建比赛菌株—公开酵母基因组映射与第一版简单基因组特征  
工作模式：本地完成，未操作 Git  
commit：无

## 修改/新增文件

- `external_data/genome/strain_mapping.csv`
- `external_data/genome/source_manifest.json`
- `external_data/genome/mapping_evidence.md`
- `external_data/genome/genome_feature_schema.json`
- `external_data/genome/strain_features.npz`
- `external_data/genome/strain_feature_index.csv`
- `external_data/genome/raw_resource_manifest.json`
- `external_data/genome/GO_OR_PATHWAY_MAPPING.csv`
- `scripts/build_genome_features.py`
- `tests/test_genome_features.py`
- `reports/GENOME_FEATURE_REPORT.md`

## 实际运行的数据

仅读取两份 metadata 的 `Strains` 与 `split_final` 字段来盘点实体和 split：train/validation metadata 8,958 行，test metadata 4,454 行，合计 13,412 行。未读取任何比赛蛋白矩阵、蛋白标签或验证/测试蛋白真值。

自动盘点结果：

| strain | metadata 行数 | seen_splits | first_seen_split | train | validation | test |
|---|---:|---|---|---|---|---|
| BAH | 2,235 | train, val_chem_only, val_time, test_chem_only, test_time | train | 是 | 是 | 是 |
| BAI | 2,248 | val_strain_only, val_both, test_both | val_strain_only | 否 | 是 | 是 |
| CEK | 2,254 | train, val_chem_only, val_time, test_chem_only, test_time | train | 是 | 是 | 是 |
| CGD | 2,207 | train, val_chem_only, val_time, test_chem_only, test_time | train | 是 | 是 | 是 |
| CRD | 2,231 | test_strain_only, test_both | test_strain_only | 否 | 否 | 是 |
| DHY210 | 2,237 | train, val_chem_only, val_time, test_chem_only, test_time | train | 是 | 是 | 是 |

真实文件与任务中的预期情况一致。

## 外部资源、版本、许可和校验值

主来源是 [Peter et al. 2018](https://doi.org/10.1038/s41586-018-0030-5) 与 [1002 Yeast Genomes 公共文件](http://1002genomes.u-strasbg.fr/files/)。论文及未另作说明的补充材料为 CC BY 4.0；公共文件服务器未单独展示原始矩阵许可，因此没有把服务器文件默认为可再分发。完整逐文件 URL、版本、日期、大小、SHA-256 和上游 MD5 在 `raw_resource_manifest.json`。

实际用于构建的 8 个缓存文件总计约 19 MB，包括 Peter Supplementary Tables、GWAS PLINK 矩阵、copy-number、presence/absence、frameshift、两个距离矩阵与 7,796 ORF FASTA。所有文件均通过固定 SHA-256；1002 提供 MD5 的 7 个文件也与上游 `md5.txt` 一致。最大输入 `1011GWASMatrix.tar.gz` 的 SHA-256 为 `4fb9ae6c2d24f169c522b582e79b81e76e436f34367df1f749ced715fbdc746a`。

S288C 坐标冻结为 `R64-1-1 / GCF_000146045.2`，Peter 群体变异也映射到 R64-1-1。当前 NCBI/SGD 注释可更新，本轮仅记录 `SGD R64-5-1 (2026-01-23)`，不用于 V1 坐标或聚合，避免把动态注释静默混入 2018 矩阵。

## 核心映射结果

| competition | external sample | mapping_type | confidence | 有效 V1 特征 |
|---|---|---|---|---|
| BAH | SX3 | supported | high | 是 |
| BAI | BJ6 | supported | high | 是，仅冻结变换 |
| CEK | JCM_2985-4B | supported | medium | 是 |
| CGD | UCD_09-448 | supported | medium | 是 |
| CRD | FIMA_3 | supported | medium | 是，仅冻结变换 |
| DHY210 | S288C / GCF_000146045.2 | proxy | low | 是，代理且不参与拟合 |

五个公开矩阵菌株保守标为 `supported`，没有声称 `exact`：Peter 表提供标准名—样本映射，第二来源闭合外部身份，但没有取得比赛提供方的一对一声明。BAI 与 CRD 的独立审计见 `mapping_evidence.md`。

## Main 只读验收后的定点修复

本轮未重新下载大型 gVCF，未修改模型、公共合同或其他线程文件。修复前后关键值：

| 指标 | 修正前 | 修正后 |
|---|---:|---:|
| 主 `genome_features` 维度 | 28 | 24 |
| DHY210 主特征有效维度 | 24 | 2 |
| DHY210 最大绝对 z 值 | 148.7967 | 0.4522 |
| 全部实体最大绝对 z 值 | 148.7967 | 3.2848 |
| 训练中心化矩阵秩 | 2 | 2 |
| PCA 维度 | 2 | 2 |

DHY210 现在只有 `assembly_total_bp` 与 `assembly_gc_percent` 进入主特征并标为有效，来源均为 NCBI `GCF_000146045.2 / R64` 装配记录。`assembly_contig_count=17` 与 `assembly_max_contig_bp=1,531,933` 也有 NCBI 装配来源，但它们属于装配 QC，故只保留在 raw/QC 工件，不进入主特征。`ploidy_n` 没有取得适用于本任务 S288C 代理培养物的冻结权威来源，已改为无效。

Peter pangenome、copy-number、frameshift 与 GWAS 矩阵都没有 S288C 行/列，因此其 called count 和所有依赖的 count/fraction/burden 对 DHY210 全部 `mask=false`。train-only 中位数仍用于执行冻结变换，但所有最终 `feature_valid_mask=false` 单元会在变换后强制数值为 0，不再暴露填充值的 z-score。

外部 ID 语义同步修正：五个公开菌株的 `external_isolate_id` 现为 SX3/BJ6/JCM_2985-4B/UCD_09-448/FIMA_3，`external_matrix_id` 才是 BAH/BAI/CEK/CGD/CRD；DHY210 为 `external_isolate_id=S288C`、`external_assembly_id=GCF_000146045.2`。

## 特征维度与各组成部分

主交付 `genome_features` 为 `6 × 24` float32，保留低维、可解释的标准化生物特征；对应 28 维未标准化 raw 数值、4 维独立技术 QC、逐维 validity、scaler 与 PCA 参数一并保存在 NPZ。

- 6 个 SNP 概览：全基因组 SNP count、Peter 的 SNP/kb、纯合/杂合 SNP、杂合比例、singleton 数。
- 4 个装配/核型生物概览：总 bp、GC%、ploidy、非整倍体片段数；其中 ploidy 与非整倍体字段对 DHY210 无效。
- 6 个 pangenome copy/presence 概览：7,708 个有一致 call 的 ORF 数、缺失数/率、多拷贝数/率、copy-number 绝对偏离负担。
- 3 个 frameshift 概览：6,575 个有 call 的基因数、frameshift 基因数/率。
- 5 个 PLINK GWAS 概览：常染色体高 MAF marker 的 called、BIM allele1 homozygous、heterozygous、missing 及 allele1 dosage fraction；不把 BIM allele1 冒充 VCF ALT。

数值分母没有偷换：`assembly_total_bp` 不是 callable bp；callable region 未获得，schema 明确记为 `not_available`。同理，没有用零伪装 insertion/deletion、染色体分担、coding/promoter、per-gene SNP/indel 或 GO 特征。

以下四个技术/装配质量字段已从主模型特征、scaler 与 PCA 中移除，只保留为 `qc_features`：`mean_read_coverage_x`、`assembly_n50_bp`、`assembly_contig_count`、`assembly_max_contig_bp`。

Copy-number/presence/frameshift 只在同一 Peter 口径覆盖五个目标 public isolate 时启用。原始 pangenome 和 frameshift 列顺序的 SHA-256 均冻结在 schema。

## 坐标与变异口径

- 参考：S288C R64-1-1；常染色体为 1–16；PLINK 的 17（线粒体）不进入 autosomal GWAS 汇总。
- Peter 全群体调用以 S288C R64-1-1 为参考。由于 V1 未下载 5.4 GB 完整 gVCF，PLINK 数字 allele 只能按源编码解释，不能宣称已独立恢复 REF/ALT。
- PLINK missing code `01` 从 called denominator 排除；heterozygous code `10` 贡献一个 dosage 单位。
- 高 MAF GWAS 矩阵为 biallelic；multiallelic、完整 insertion/deletion、repeat/low-quality 再过滤不从派生矩阵反推。
- 基因坐标、promoter 与多转录本聚合没有在缺少“完整 VCF + 已验证坐标注释”的情况下拼接；这符合 fail-closed。

## PCA 拟合菌株、矩阵秩和实际维数

拟合 mask 严格为：

```text
seen_in_train=true
AND mapping_type in {exact, supported}
AND feature_valid=true
```

实际拟合菌株只有 `BAH, CEK, CGD`。`BAI, CRD, DHY210` 均不参与中位数、scaler、方差筛选或 PCA。修正后训练中心化矩阵秩为 2，实际 PCA 为 2 维，解释方差比约 `0.6461, 0.3539`。由于只有三株可靠训练菌株，PCA 标为辅助表示，主交付仍是 24 维可解释标准化特征。

## correct/shuffle/zero 接口

`apply_feature_variant` 返回变体矩阵与“目标行 → 来源行”的完整 permutation。`shuffle` 只在 `exact/supported AND feature_valid` 之间重排，proxy/unresolved/invalid 保持原位；特征多重集不变。`zero` 只清零数值矩阵，不改变外部 index、mapping type、confidence 或 validity。默认 seed 为 20260814，测试覆盖多个 seed 的确定性行为，但本任务未训练模型，也未报告消融性能。

## 测试命令和结果

```powershell
$env:PYTHONPATH='D:\虚拟细胞\tmp\python_pkgs'
D:\虚拟细胞\.venv\Scripts\python.exe -m pytest D:\虚拟细胞\tests\test_genome_features.py -q
```

最终结果：`20 passed`。除原 16 项外，新增行为测试验证：无效分母会使全部派生量无效；所有最终无效特征严格为 0；DHY210 不再出现缺失填充导致的极端 z 值；isolate/matrix/assembly ID 不混淆；四个 QC 字段不进入主特征；scaler/PCA 仍只由 BAH、CEK、CGD 拟合。

## 与基线比较

不适用，本任务不训练模型，也不填写 validation、shuffle 或 zero 性能。

## 失败或未完成项目

1. 完整 `1011Matrix.gvcf.gz` 约 5.4 GB，本轮没有下载；因此 indel、chromosome、gene/promoter、callable-bp 负担保持 `not_available`。
2. 官方 SGD GO Slim 下载在本机网络环境中两次未完整完成；所有截断文件均删除，没有进入 manifest 的有效资源清单。`GO_OR_PATHWAY_MAPPING.csv` 是一行显式 `not_available` 状态，不是假映射。
3. SGD 历史压缩包及 NCBI 固定装配 FASTA 的有界下载均在本机网络环境中截断，截断文件已删除。DHY210 代理只使用 NCBI/SGD 公布的 R64 装配事实和参考相对零负担，不声称本地 FASTA 已验证。
4. `.xls` 读取依赖 `xlrd 2.0.1`。本机临时依赖位于 `tmp/python_pkgs`，不是交付文件；Main 需要在正式环境固定该版本或先导出所需小表。
5. 1002 原始服务器的矩阵再分发条款没有单独明示，Main 在对外分发缓存前应做许可审查；派生小特征和出处已完整记录。

是否涉及接口变更：否。没有修改 `project_v2` 公共合同；交付遵循现有 `strain_id/external_isolate_id/mapping_type/mapping_confidence/genome_features/feature_valid_mask` 约定。

需要 Main 决定的问题：是否批准后续下载完整 gVCF 与 SGD 坐标注释/GO Slim，并在许可复核后补齐 indel、callable、gene/promoter 与 GO 负担；是否接受 `supported` 映射进入主模型，尤其 CEK/CGD/CRD 的 medium 置信度；是否在正式环境固定 `xlrd==2.0.1`。

建议下一步：先由 Main 人工复核 `mapping_evidence.md` 中 BAI/CRD 与五个 `supported` 结论；批准后再做独立 V2.1 大资源扩展，不要在当前 28 维简单特征验收前引入 Genome LM。

## 逐菌株报告

BAH：`supported/high`；external isolate ID `SX3`，matrix ID `BAH`。Peter S1 明确配对，NCBI 与同行评议表独立复核；已生成有效特征，参与 train-only 拟合。

BAI：`supported/high`；external isolate ID `BJ6`，matrix ID `BAI`。Peter S1、NCBI 与 ScRAPdb 闭合证据；已生成有效特征，但不参与任何拟合，只执行冻结变换。

CEK：`supported/medium`；external isolate ID `JCM_2985-4B`，matrix ID `CEK`。Peter S1 与独立学术复分析表闭合；已生成有效特征，参与 train-only 拟合。

CGD：`supported/medium`；external isolate ID `UCD_09-448`，matrix ID `CGD`。Peter S1 与两份独立学术表复核；已生成有效特征，参与 train-only 拟合。

CRD：`supported/medium`；external isolate ID `FIMA_3`，matrix ID `CRD`。Peter S1 与同行评议研究复核；已生成有效特征，但 CRD 仅在 test 出现，只执行冻结变换。

DHY210：`proxy/low`；external isolate ID `S288C`，assembly ID `GCF_000146045.2`。这是显式代理而非等价菌株，会丢失 DHY210 特异变异；主特征仅 2 个来源明确的维度有效，`proxy_flag=true`，不参与拟合。
