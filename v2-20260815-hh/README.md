# GOAI 酵母虚拟细胞 V2

这是 V2 最终候选模型的可复核发布包。它包含模型源码、冻结外部特征、六个专家 checkpoint、配置、测试和阶段报告；不包含比赛原始数据、测试集蛋白真值、生成出的 193 MB 提交表，也不包含未进入最终模型且许可受限的研究数据。

## 模型是什么

最终候选是一个双随机种子、按样本新颖性路由的三专家集成：

- `D0`：对新化合物采用保守预测，不使用未经证实的化学迁移。
- `S1 C`：处理已见化合物、未见菌株且技术批次已知的样本。
- `D2`：处理已见化合物与菌株，并提供层级批次回退。
- 每个专家使用 seed `20260814` 与 `20260815` 的预测等权平均。
- Quality Control、Water、DMSO 统一走 D0，并在推理层强制药物响应为零。

模型仍有明确局限：当前 Morgan 化学表征没有在所有新药场景稳定获益，最终策略因此对真正新药采取保守路由；FC 训练使用的是明确标注为非官方的 `pert_id` 奇偶映射实验口径，不能作为官方 FC 结果解释。

## 目录

```text
v2/
├── baseline/                  # 模型、训练、评估、路由和自动测试
├── external_data/
│   ├── chemistry/            # PubChem 映射、RDKit/Morgan 冻结特征及缓存
│   └── genome/               # 6 个菌株的冻结派生特征与映射证据
├── project_v2/data_contract/ # 4,422 蛋白顺序、控制匹配和审计合同
├── reports/                   # 关键阶段报告、指标、checkpoint 与配置
├── scripts/                   # 特征构建、评分审计、便携推理和包校验
├── tests/                     # 外部特征与发布包测试
├── docs/                      # 架构、数据许可和复现说明
└── WAYB_WAYC/                # 仅放置说明；比赛数据由使用者自行复制
```

## 环境

已验证环境为 Python 3.12.13、PyTorch 2.11.0+cu128、NumPy 2.3.5、Pandas 3.0.1、RDKit 2026.03.5。安装基础依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

若需 CUDA，请按本机 CUDA 环境从 PyTorch 官方渠道安装对应构建，不要盲目复制 `+cu128` 后缀。

## 放置比赛 metadata

推理只需要两份 metadata，不需要任何 proteome 真值。将以下文件自行复制到 `WAYB_WAYC/`：

```text
WAYB_WAYC_metadata_train_val(1).csv
WAYB_WAYC_metadata_test(1).csv
```

比赛数据受比赛规则约束，已被 `.gitignore` 排除，不能提交 GitHub。当前冻结模型只适用于本项目使用的官方数据快照：train/validation metadata 8,958 行，test metadata 4,454 行。

## 生成预测

在 `v2` 根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py --root . --output-dir outputs\run1 --device auto
```

输出目录必须事先不存在。主要输出为：

- `prediction.csv`：4,454 × 4,423，log2 尺度；
- `routing_manifest.csv`：每个测试样本选择哪个专家及原因；
- `pass_manifest.json`：哈希、设备、耗时和防泄漏审计；
- 两个 float32 数组，用于复核预测和响应。

生成过程预计需要数百 MB 临时磁盘空间。不要将输出目录提交 GitHub。

## 校验

```powershell
.\.venv\Scripts\python.exe scripts\verify_package.py --root .
.\.venv\Scripts\python.exe -m pytest tests\test_package_layout.py baseline\tests\test_model_v2_losses.py -q
```

`PACKAGE_MANIFEST.json` 保存包内文件 SHA-256。六个最终专家 checkpoint 和三份冻结合同另有固定哈希检查。

放置比赛 metadata 后，可再运行 `baseline/tests/test_model_v2_contract.py` 中依赖真实 metadata 的完整合同测试。

## 结果口径

最终提交前验证重放结果为四场景平均 RMSE `0.638989552855`、平均 raw-FC PCC `0.304467385473`。这些是规划代理与本地验证结果，不是官方榜单分数。完整限制和证据见 [MODEL_CARD.md](docs/MODEL_CARD.md) 与 `reports/MODEL_V2_STAGE_S3_FINAL_PRESUBMISSION_AUDIT.md`。

## 发布前仍需决定

本包没有替团队选择源码许可证。公开仓库若希望他人不仅“查看”，还可以合法复用或修改，需要所有权人选择并添加许可证（如 MIT 或 Apache-2.0）。第三方数据许可不随源码许可证改变，详见 [THIRD_PARTY_DATA.md](docs/THIRD_PARTY_DATA.md)。
