# 授权比赛 metadata 放置说明

本目录在仓库中只能保留本说明。请从组委会授权渠道取得数据，并在本地放置：

```text
WAYB_WAYC_metadata_train_val(1).csv
WAYB_WAYC_metadata_test(1).csv
```

最终推理不需要、也不应在本目录放置任何 proteome 文件。`.gitignore` 会排除 metadata，但上传前仍须人工检查 Git 暂存区。
