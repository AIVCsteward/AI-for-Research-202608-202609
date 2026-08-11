# AutoDL 上传与运行指南

## 1. 租用 AutoDL 服务器

1. 打开 [AutoDL](https://www.autodl.com)，注册/登录
2. 租用一台 GPU 实例，推荐配置:
   - **GPU**: RTX 3090 / 3080 Ti / A4000（够用且便宜）
   - **镜像**: PyTorch 2.0+ 官方镜像
   - **数据盘**: 20GB+（数据 ~420MB 压缩）

## 2. 上传文件到 AutoDL

将以下文件/文件夹打包上传到 AutoDL 实例:

```
项目根目录需要包含:
├── baseline/          ← 核心模块
├── aivc/              ← 高级模型模块
├── experiments/       ← 消融实验模块
├── data/              ← 数据 (WAYB_WAYC_*.csv)
├── run_full_pipeline.py  ← 一键运行脚本
└── requirements_autodl.txt
```

**推荐方法 1**: 用 AutoDL 的「文件上传」功能，先打包成 zip 再上传
**推荐方法 2**: 用 AutoDL 的 JupyterLab 界面拖拽上传
**推荐方法 3**: 用 `scp` 上传（AutoDL 提供 SSH 连接信息）

```bash
# 本地打包 (排除不需要的文件)
cd /path/to/AIVC
zip -r aivc_upload.zip \
    baseline/ aivc/ experiments/ data/ \
    run_full_pipeline.py requirements_autodl.txt \
    -x "*/__pycache__/*" "*.pyc" ".git/*"
```

## 3. AutoDL 终端操作

```bash
# 1. 进入项目目录
cd /root/autodl-tmp/  # 或其他你上传的目录

# 2. 解压 (如果上传的是 zip)
unzip aivc_upload.zip

# 3. 安装依赖
pip install -r requirements_autodl.txt

# 4. 验证环境
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 5. 运行全流程实验!
python run_full_pipeline.py --epochs 100 --output-dir experiments/outputs
```

## 4. 运行参数

```bash
# 完整运行 (全部 5 组实验, 约 20-30 min on GPU)
python run_full_pipeline.py --epochs 100

# 快速测试 (减少 epoch)
python run_full_pipeline.py --epochs 20

# 只跑特定实验
python run_full_pipeline.py --epochs 100 --skip gnn loss    # 跳过 GNN 和 loss 消融
python run_full_pipeline.py --epochs 100 --skip encoder arch loss  # 只跑主模型

# 指定 GPU
python run_full_pipeline.py --epochs 100 --device cuda:0
```

## 5. 结果

运行完成后在 `experiments/outputs/full_pipeline/` 下生成:

```
FULL_REPORT.md          ← 完整的 Markdown 报告 (可直接查看)
all_results.json        ← 结构化 JSON 结果
prediction_main.csv     ← 主模型提交文件
prediction_gnn.csv      ← GNN 模型提交文件
main_model.pt           ← 主模型 checkpoint
gnn_model.pt            ← GNN 模型 checkpoint
```

## 6. 下载结果

在 AutoDL 的 JupyterLab 中右键下载 `FULL_REPORT.md` 和 `all_results.json`。

---

## 备用: 本地 CPU 运行

如果暂时不用 AutoDL，也可以在你笔记本上跑:

```bash
# 快速验证 (5 epochs)
python run_full_pipeline.py --epochs 5

# 完整运行 (约 2 小时)
python run_full_pipeline.py --epochs 100
```
