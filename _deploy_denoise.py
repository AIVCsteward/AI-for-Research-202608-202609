"""上传去噪脚本到 AutoDL，生成缓存+去噪目标，启动 AIVCModel 多 k 训练。"""
import paramiko

HOST = "connect.nmb2.seetacloud.com"
PORT = 45188
USER = "root"
PWD = "RFc7c3KwA+sZ"
LOCAL_TAR = "d:/AIVC/_deploy_denoise.tar.gz"
REMOTE_TAR = "/root/autodl-tmp/_deploy_denoise.tar.gz"
PY = "/root/miniconda3/bin/python"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)

# 1. 上传
sftp = c.open_sftp()
sftp.put(LOCAL_TAR, REMOTE_TAR)
sftp.close()
print("[1/4] 上传完成")

# 2. 解压 + 验证
cmd = (
    f"cd /root/autodl-tmp && tar xzf _deploy_denoise.tar.gz && rm _deploy_denoise.tar.gz && "
    "echo UNPACKED && grep -c denoised_fc_target aivc/training.py"
)
stdin, stdout, stderr = c.exec_command(cmd, timeout=60)
print("[2/4] 解压验证:", stdout.read().decode().strip())

# 3. 生成缓存 + 去噪目标（同步等待，约 5 分钟）
cmd = (
    f"cd /root/autodl-tmp && {PY} -u -m scripts.precompute_oracle_cache && "
    f"{PY} -u -m scripts.build_denoise_targets"
)
print("[3/4] 生成缓存 + 去噪目标（约 5 分钟）...")
stdin, stdout, stderr = c.exec_command(cmd, timeout=600)
out = stdout.read().decode("utf-8", "replace")
err = stderr.read().decode("utf-8", "replace")
print(out[-1500:])
if err.strip():
    print("STDERR:", err[-1500:])

# 4. 启动训练（setsid 后台）
cmd = (
    f"cd /root/autodl-tmp && setsid {PY} -u -m experiments.run_denoise_ablation "
    "--epochs 50 --model aivc --device cuda "
    "> denoise_aivc_k.log 2>&1 < /dev/null & echo LAUNCHED"
)
stdin, stdout, stderr = c.exec_command(cmd, timeout=30)
print("[4/4] 启动训练:", stdout.read().decode().strip())

c.close()
