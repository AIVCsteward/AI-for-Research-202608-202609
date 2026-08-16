# GOAI 虚拟细胞开放知识榜提交包

本目录是“对照锚定、按新颖性路由的双随机种子三专家集成模型”的最小复现包，面向 GOAI 赛道三虚拟细胞方向的**开放知识榜**。包内保留最终推理源码、六个 Stage B 专家 checkpoint、四个 Stage A 词表/锚点 checkpoint、冻结外部特征、配置、来源披露和自动校验；不包含比赛原始 metadata、训练/验证/测试蛋白质组真值、样本级路由清单或生成后的 `prediction.csv`。

## 1. 模型与路由

- D0：新化合物、Water、DMSO 和 Quality Control 使用保守专家；对 control/QC 强制 `delta_response=0`。
- S1 C：已见化合物、未见菌株且技术批次 tuple 已知时使用 flat-batch 专家。
- D2：已见化合物与菌株使用 hierarchical-batch 专家；技术字段按 `data_source → instrument → plate` 分层回退。
- 每个专家分别使用随机种子 `20260814` 和 `20260815`，推理时按 `0.5/0.5` 等权平均。
- 输出为固定顺序的 4,422 维 log2 蛋白质组预测。

模型使用的公开外部知识包括 PubChem/RDKit 化学结构特征和公开酵母基因组派生特征，因此只能申报开放知识榜。详细来源见 [OPEN_KNOWLEDGE_DISCLOSURE.md](docs/OPEN_KNOWLEDGE_DISCLOSURE.md)。

## 2. 数据边界

比赛数据不随本包分发。使用者必须从组委会授权渠道取得数据，并把两份 metadata 放入 `WAYB_WAYC/`：

```text
WAYB_WAYC_metadata_train_val(1).csv
WAYB_WAYC_metadata_test(1).csv
```

最终推理不需要任何 proteome 文件。训练、蛋白筛选、归一化和对照统计仅允许使用 `split_final=train` 蛋白标签；validation 只用于模型选择与报告；测试蛋白真值不得进入推理、特征或统计量。

## 3. 安装

已验证环境为 Python 3.12.13。基础安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

CUDA 版 PyTorch 应按目标机器的 CUDA 环境从 PyTorch 官方渠道安装。

## 4. 生成提交文件

在本目录运行：

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py `
  --root . `
  --output-dir outputs\final `
  --device auto
```

输出目录必须事先不存在。主要产物：

- `prediction.csv`：4,454 行，第一列为 `sample_ID`，其余为 4,422 个蛋白；
- `routing_manifest.csv`：仅生成在本地输出目录，不应上传公共仓库；
- `pass_manifest.json`：记录设备、哈希、路由计数和防测试真值访问结果；
- `prediction_float32.npy` 与 `response_float32.npy`：复核用中间结果。

预测保持 log2 尺度，不做指数还原、裁剪、重新缩放或提交后校准。

## 5. 上传前校验

不放置比赛 metadata 也能执行：

```powershell
.\.venv\Scripts\python.exe scripts\verify_submit_package.py --root .
.\.venv\Scripts\python.exe -m pytest tests -q
```

放置授权 metadata 后，再执行一次真实推理并核对：行数、列数、蛋白顺序、无 NA/inf、`test_proteome_opened=false`。

## 6. 规则口径说明

- 8 月补充材料中的 4,232 与当前官方数据快照按 train-only 80% 缺失规则得到的 4,422 不一致。本包冻结 4,422，并与当前参考 prediction 表头完全一致；正式提交仍须以平台最后发布的 feature contract 为准，不能为了匹配教程文字手工删列。
- 训练阶段使用由 train 数据冻结的本地 control 选择规则。组委会评分端独立按统一 DMSO/Water 规则计算评分，参赛队无须复现评测端映射。
- 当前包不包含官方榜单成绩。本地 validation 结果不能写成正式测试分数。

## 7. 发布边界

本包已删除样本级配对、holdout ID、validation routing 和历史失败实验。冻结实体特征索引仍包含推理所必需的比赛实体名称；在组委会未明确允许公开这些派生内容前，只应上传到团队私有仓库或提交给主办方复现审核。

团队尚未选择源码许可证。公开发布前必须处理 [LICENSE_DECISION_REQUIRED.md](LICENSE_DECISION_REQUIRED.md)。
