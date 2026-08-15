# 复现说明

## 推理复现

1. 安装 `requirements.txt`。
2. 按 `WAYB_WAYC/README.md` 放置两份 metadata。
3. 运行 `python scripts/verify_package.py --root .`。
4. 运行 `python scripts/run_inference.py --root . --output-dir outputs/run1 --device auto`。
5. 对照 `pass_manifest.json` 中的行列数、路由计数、文件哈希和 `test_proteome_opened=false`。

不放置比赛 metadata 时，只运行发布包布局测试和纯模型/损失测试；真实 control anchor、实体 shuffle 等合同测试会因缺少 metadata 主动失败，这是数据未随包分发造成的预期行为。

推理依赖的六个 Stage B checkpoint 和四个 Stage A checkpoint保留在原验证路径下，以避免改变经过测试的路径合同。

## 外部特征重建

化学特征：

```powershell
python scripts/build_chemical_features.py --help
```

菌株特征：

```powershell
python scripts/build_genome_features.py --help
```

菌株原始缓存未再分发；URL、版本和 SHA-256 位于 `external_data/genome/raw_resource_manifest.json`。重建时必须自己下载并核验。

## 训练复现边界

训练入口和全部正式配置保存在 `baseline/`，但本发布包不包含比赛蛋白训练标签，因此不能脱离有授权的比赛数据重新训练。训练时必须遵守 train-only 统计，验证/测试标签不得用于梯度、特征或标准化器。

## 历史结果

关键阶段的 Markdown、CSV、JSON、history、resolved config 和 checkpoint 已保留。失败的 NetwoRx 外部功能预训练数据没有进入发布包；其结论可在团队内部原工作区查阅，但不属于最终推理依赖。
