from pathlib import Path
import paramiko
pw = Path('.codex/gpu_password.txt').read_text(encoding='utf-8').strip()
t = paramiko.Transport(('connect.nmb2.seetacloud.com', 32942)); t.connect(); t.auth_password('root', pw)
c = paramiko.SSHClient(); c._transport = t
cmd = '/root/miniconda3/bin/python -m pip install -q huggingface_hub && /root/miniconda3/bin/hf download Qwen/Qwen3-1.7B --local-dir /root/models/Qwen3-1.7B && du -sh /root/models/Qwen3-1.7B && df -h /root'
_i, o, e = c.exec_command(cmd, timeout=1800)
text = o.read().decode(errors='replace') + e.read().decode(errors='replace')
Path('benchmark-results/gpu/download.log').write_text('$ ' + cmd + '\n' + text, encoding='utf-8')
print(text)
