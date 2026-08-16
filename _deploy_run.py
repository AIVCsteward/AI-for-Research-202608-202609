"""Temporary helper: upload + extract updated source, then launch AIVCModel ablation."""
import paramiko

HOST = "connect.nmb2.seetacloud.com"
PORT = 45188
USER = "root"
PWD = "RFc7c3KwA+sZ"
LOCAL_TAR = "d:/AIVC/_aivc_src.tar.gz"
REMOTE_TAR = "/root/autodl-tmp/_aivc_src.tar.gz"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)

sftp = c.open_sftp()
sftp.put(LOCAL_TAR, REMOTE_TAR)
sftp.close()
print("uploaded")

cmd = (
    "cd /root/autodl-tmp && tar xzf _aivc_src.tar.gz && rm _aivc_src.tar.gz && "
    "grep -c monitor_fn aivc/training.py && grep -c 'model_type' experiments/run_encoder_ablation.py && "
    "echo SYNCED && "
    "setsid /root/miniconda3/bin/python -u -m experiments.run_encoder_ablation "
    "--epochs 50 --model aivc --use-residual-loss "
    "--groups full no_chemical_structure "
    "--output-dir experiments/outputs/chem_ablation_aivc "
    "> chem_ablation_aivc.log 2>&1 < /dev/null & echo LAUNCHED"
)
stdin, stdout, stderr = c.exec_command(cmd, timeout=60)
try:
    print(stdout.read().decode("utf-8", "replace"))
except Exception as e:
    print("(stdout read timeout — launch likely still fired)")
c.close()
