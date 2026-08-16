# 第三方软件依赖

| 软件 | 冻结版本 | 用途 | 上游许可证/项目页 |
|---|---:|---|---|
| Python | 3.12.13 | 运行环境 | PSF License，https://www.python.org/ |
| PyTorch | 2.11.0 | 模型与推理 | BSD-style，https://pytorch.org/ |
| NumPy | 2.3.5 | 数值计算与 NPZ 工件 | BSD-3-Clause，https://numpy.org/ |
| pandas | 3.0.1 | metadata/CSV 对齐与提交生成 | BSD-3-Clause，https://pandas.pydata.org/ |
| RDKit | 2026.03.5 | 分子结构、Morgan 与描述符 | BSD-3-Clause，https://www.rdkit.org/ |
| Requests | 2.34.2 | PubChem PUG-REST 获取 | Apache-2.0，https://requests.readthedocs.io/ |
| pytest | 9.1.1 | 自动测试 | MIT，https://pytest.org/ |
| xlrd | 2.0.1 | 读取 Peter 2018 `.xls` 补充表 | BSD，https://xlrd.readthedocs.io/ |

上表用于复现披露，不替代各项目正式许可证文本。CUDA、显卡驱动和操作系统不随包分发；使用者应根据目标硬件自行安装兼容版本。
