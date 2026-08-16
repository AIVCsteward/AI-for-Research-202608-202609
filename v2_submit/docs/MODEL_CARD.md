# 模型卡

## 用途

本模型用于 GOAI 虚拟细胞开放知识榜指定数据快照的酵母扰动蛋白质组条件响应预测。它不是临床、药物安全或机制因果推断工具。

## 输入与输出

- 输入：菌株、化合物、培养基、温度、处理时间以及允许的技术批次 metadata；化合物和菌株同时关联公开外部特征。
- 输出：固定 feature contract 顺序的 4,422 维 log2 蛋白强度。
- 最终恒等式：`y_pred = y_baseline + delta_response + delta_batch`。

## 最终候选

- 三专家：D0、S1 C、D2。
- 随机种子：20260814、20260815。
- 双 seed 权重：0.5 / 0.5。
- 路由只依赖 metadata 及 train 中是否已见，不使用 validation/test 蛋白标签。
- Water、DMSO、Quality Control 强制使用 D0，且 response 为零。

## 验证结论

本地验证重放的四场景平均 RMSE 为 0.638989552855，平均 raw-FC PCC 为 0.304467385473。它们是本地 validation 结果，不是官方榜单成绩。

## 已知局限

- 真正新化合物的结构到蛋白响应迁移仍不稳定，因此 D0 对新药采取保守预测。
- Matched Control 的绝对保真度仍然很强，模型没有在全部评分模块稳定超过它。
- 公开 validation/test 没有真正未见的技术 batch tuple，独立内部新批次仍可能造成性能下降。
- DHY210、Oligomycin 和 Tunicamycin 存在已披露的代理或身份不确定性。

## 防泄漏

训练统计只接受 `split_final=train` ID；测试推理入口有 test proteome 文件名访问守卫；外部特征标准化器只用训练实体拟合。
