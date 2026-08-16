# 开放知识榜外部资源披露

## 最终模型实际使用

| 资源 | 用途 | 版本/日期 | 许可或使用边界 | 可复现位置 |
|---|---|---|---|---|
| PubChem PUG-REST | 化合物 CID、SMILES、InChIKey、分子式与名称证据 | 响应获取日期 2026-08-13 | 遵循 NCBI/PubChem 政策 | `external_data/chemistry/source_manifest.json`、`pubchem_cache/` |
| RDKit | Morgan fingerprint、分子描述符、结构标准化 | 2026.03.5 | BSD-3-Clause | `scripts/build_chemical_features.py` |
| Peter et al. 2018 | 比赛菌株到 1011 项目菌株的映射与群体基因组派生特征 | Nature 556:339-344 | 论文及补充材料 CC BY 4.0 | `external_data/genome/source_manifest.json` |
| 1002/1011 Yeast Genomes 公共文件 | SNP、拷贝数、存在缺失及距离派生特征 | 2018 数据快照 | 原始服务器未单列再分发条款；原始缓存不随包分发 | `external_data/genome/raw_resource_manifest.json` |
| NCBI S288C R64 | DHY210 代理的参考组装信息 | GCF_000146045.2 | 遵循 NCBI 数据政策 | `external_data/genome/source_manifest.json` |

## 身份与 fallback

- DHY210 使用 S288C 作为 `proxy/low`，不声明精确等价。
- Oligomycin 使用 Oligomycin A 作为 `proxy/medium`。
- Tunicamycin 保持 unresolved，使用显式 fallback。
- Cisplatin 与 NaCl 身份确认，但当前普通有机分子 Morgan/RDKit 特征不适用，使用显式零特征。
- 化学 descriptor scaler 仅由训练化合物拟合；基因组 scaler/PCA 仅由训练菌株拟合。

## 未进入最终模型

NetwoRx、Parsons、Hillenmeyer、MOSAIC、HIP/HOP 仅用于内部覆盖率或失败路线诊断，没有进入最终模型、checkpoint 或本提交包。GO/KEGG/PPI 也未接入最终推理，不应在申报材料中写成已经使用。

## 商业服务与闭源模型

最终模型没有调用商业 API，也没有使用闭源模型。PubChem 是公开数据服务；最终推理完全离线读取冻结工件。
