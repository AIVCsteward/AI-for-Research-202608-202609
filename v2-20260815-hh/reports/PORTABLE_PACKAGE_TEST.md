# V2 便携发布包验证

验证日期：2026-08-15。

## 测试范围

- 推理工作根目录切换到独立的 `v2/` 发布目录。
- 临时提供两份授权 metadata；不提供任何 proteome 文件。
- 只使用包内冻结外部特征、四个 Stage A checkpoint 和六个 Stage B checkpoint。
- CUDA 完整推理 4,454 个样本、4,422 个蛋白。

## 结果

- 状态：PASS。
- 专家路由：D0 2,997；S1 C 1,322；D2 135。
- `prediction.csv` SHA-256：`9739f087788bfd37f57750564c9d351bbf37bd867ab7e87820294eec52c8dbe9`。
- 该哈希与 Stage S3 最终候选完全一致。
- 警告与错误：0。
- test proteome 打开：false。
- 完整推理耗时：约 3.76 秒（RTX 5060 Laptop GPU；不含 Python 进程启动和文件清理）。

测试后已删除临时 metadata、副本预测和 Python 缓存；它们不属于发布包。
