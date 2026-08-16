# GOAI 虚拟细胞菌株映射与基因组特征包

本包提供 GOAI 虚拟细胞赛题六个菌株到公开酵母菌株的可复核映射，以及模型可直接读取的冻结基因组特征。它包含官方比赛 metadata 的专用接入脚本，可把每个 `sample_ID` 自动转换成对应的菌株特征和有效性掩码。包内不含比赛原始数据，也不含体积较大的上游基因组缓存。

## 映射范围

| 比赛菌株 | 公开菌株 | 映射状态 | 置信度 |
|---|---|---|---|
| BAH | SX3 | supported | high |
| BAI | BJ6 | supported | high |
| CEK | JCM_2985-4B | supported | medium |
| CGD | UCD_09-448 | supported | medium |
| CRD | FIMA_3 | supported | medium |
| DHY210 | S288C | proxy | low |

`supported` 表示公开来源支持该对应关系，但没有比赛数据提供方的一对一身份声明。DHY210 使用 S288C 工程代理，不代表二者基因组等价。

## 目录说明

- `external_data/genome/strain_mapping.csv`：完整映射、证据摘要、来源和置信度。
- `external_data/genome/strain_feature_index.csv`：模型行索引及菌株顺序。
- `external_data/genome/strain_features.npz`：6×24 冻结主特征、逐维有效性掩码、原始特征、QC 特征和 PCA 工件。
- `external_data/genome/genome_feature_schema.json`：特征定义、顺序、标准化规则和 fallback 合同。
- `external_data/genome/mapping_evidence.md`：逐菌株人工复核证据。
- `external_data/genome/source_manifest.json`：派生工件来源和生成信息。
- `external_data/genome/raw_resource_manifest.json`：上游资源 URL、版本、许可和校验值。
- `scripts/load_strain_features.py`：无需比赛标签的直接读取示例。
- `scripts/prepare_competition_genome_inputs.py`：将官方比赛 metadata 转成逐样本模型输入。
- `scripts/build_genome_features.py`：从允许的上游资源重建特征的脚本。
- `competition_contract.json`：比赛文件名、字段、样本数及 split 快照合同。
- `WAYB_WAYC/README.md`：官方文件放置说明；目录中不附带比赛原始数据。
- `docs/COMPETITION_INTEGRATION.md`：比赛模型接入说明及输出定义。
- `tests/test_packaged_artifacts.py`：不依赖比赛原始数据的轻量测试。
- `tests/test_competition_integration.py`：比赛 metadata 接口的合成数据测试。
- `tests/test_genome_features.py`：完整重建测试，需要比赛 metadata 和上游缓存。

## 直接读取

安装 NumPy 后，在本目录运行：

```powershell
python scripts/load_strain_features.py --strain BAI
```

Python 接口：

```python
from scripts.load_strain_features import load_strain_feature

record = load_strain_feature("BAI")
features = record["features"]
valid_mask = record["valid_mask"]
```

特征已经按照训练菌株拟合的冻结规则完成标准化。使用者不应再用验证或测试菌株重新拟合 scaler 或 PCA。

## 接入官方比赛数据

从比赛平台下载以下两份 metadata，放入本包的 `WAYB_WAYC/`：

```text
WAYB_WAYC_metadata_train_val(1).csv
WAYB_WAYC_metadata_test(1).csv
```

随后运行：

```powershell
python scripts/prepare_competition_genome_inputs.py `
  --metadata-train-val "WAYB_WAYC/WAYB_WAYC_metadata_train_val(1).csv" `
  --metadata-test "WAYB_WAYC/WAYB_WAYC_metadata_test(1).csv" `
  --output-dir outputs/competition_genome_inputs
```

输出包括逐样本特征 NPZ、逐样本映射 CSV 和可追溯 manifest。脚本只读取 metadata，不读取任何 proteome 文件。

## 快速校验

```powershell
python -m pytest tests/test_packaged_artifacts.py tests/test_competition_integration.py -q
```

## 使用边界

本包可直接覆盖 BAH、BAI、CEK、CGD、CRD、DHY210。输入其他菌株时读取接口会明确报错，不会静默替换为某个已知菌株。若需支持新的菌株，必须先取得可靠的公共样本映射，再按同一特征合同生成特征。
