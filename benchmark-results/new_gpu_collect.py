from pathlib import Path
import paramiko

pw = Path('.codex/gpu_password.txt').read_text(encoding='utf-8').strip()
t = paramiko.Transport(('connect.nmb2.seetacloud.com', 40423)); t.connect(); t.auth_password('root', pw)
c = paramiko.SSHClient(); c._transport = t
commands = {
    'env.log': "nvidia-smi; python --version; python3 --version 2>/dev/null || true",
    'torch.log': "python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())' 2>&1 || python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())' 2>&1",
    'model.log': "ls -lh /root/models/models/Qwen--Qwen3-1.7B/snapshots/master; du -sh /root/models/models/Qwen--Qwen3-1.7B/snapshots/master",
    'code.log': "ls -la /root/bench/vllm-v2-npu 2>/dev/null || true",
}
out = Path('benchmark-results/gpu-new'); out.mkdir(exist_ok=True)
for f, cmd in commands.items():
    _i,o,e=c.exec_command(cmd,timeout=180); text=o.read().decode(errors='replace')+e.read().decode(errors='replace'); (out/f).write_text('$ '+cmd+'\n'+text,encoding='utf-8'); print(text)
c.close()
