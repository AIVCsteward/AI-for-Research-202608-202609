# 第三方数据与许可边界

## 包内包含并被最终模型实际读取

### PubChem / RDKit 化学特征

- PubChem PUG-REST：动态服务，响应日期和 SHA-256 记录在 `external_data/chemistry/source_manifest.json`；使用与再分发须遵守 NCBI/PubChem 政策。
- RDKit：BSD-3-Clause；冻结生成版本 2026.03.5。
- 包内保存比赛化合物的映射、人工复核、冻结 Morgan/descriptor 特征和小型 PubChem 响应缓存。

### 公开酵母基因组派生特征

- Peter et al. 2018 文章及补充材料：CC BY 4.0。
- 1002/1011 Yeast Genomes 公共服务器文件：论文为 CC BY 4.0，但服务器原始文件未单独声明再分发条款。
- NCBI S288C R64：遵循 NCBI 数据政策。
- 因此 GitHub 包只包含最终模型读取的派生特征、映射、哈希、来源清单和重建脚本；`raw_cache/` 原始下载包不随包分发。

## 明确不包含

- GOAI 比赛 metadata/proteome 与测试真值：受比赛规则约束，由使用者自行取得。
- OfficialRules、赛题解读 PDF：仅作比赛参考，未随包再分发。
- NetwoRx、Parsons、Hillenmeyer、MOSAIC、HIP/HOP 外部功能数据：没有进入最终候选；其中部分存在派生再分发许可风险，全部排除。
- 生成的 `prediction.csv`：属于大型运行产物，不是训练依赖，且单文件超过 GitHub 普通 100 MB 限制。

## 身份披露

- DHY210 → S288C：`proxy`，不是精确等价。
- Oligomycin → Oligomycin A：`proxy/medium`。
- Tunicamycin：`unresolved`，使用显式 fallback。
- Cisplatin 与 NaCl：身份 confirmed，但冻结化学数值特征无效，使用显式 fallback。

详细逐文件来源、URL、版本、日期和哈希以两个 `source_manifest.json` 为准；这些冻结清单不能修改，否则 checkpoint 合同会失败。
