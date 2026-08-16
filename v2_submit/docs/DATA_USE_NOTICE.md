# 比赛数据使用与上传边界

## 本包不包含

- GOAI 原始 metadata；
- train/validation/test proteome；
- 测试真值与自评分数；
- 样本级 validation/test 路由；
- treatment-control 样本 ID 配对；
- batch holdout 样本 ID；
- 生成后的大型 `prediction.csv`。

## 本包包含

- train-only 冻结的蛋白 feature contract；
- 公开数据库生成的化学与基因组派生特征；
- 为运行冻结 checkpoint 所必需的实体索引；
- 六个最终专家 checkpoint 和四个 Stage A 词表/锚点 checkpoint；
- 模型、推理、特征重建和防泄漏代码。

实体索引中的比赛实体名称来自授权 metadata，是冻结推理接口的一部分。在主办方未明确允许公开派生实体表前，本包应放在私有协作仓库或直接交主办方，不能视为已经通过公共再分发许可审查。

本包的 `.gitignore` 只能防止常见误提交，不能替代人工审查。每次上传前必须运行 `scripts/verify_submit_package.py`。
