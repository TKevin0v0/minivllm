from pathlib import Path
import paramiko
pw = Path('.codex/gpu_password.txt').read_text(encoding='utf-8').strip()
t = paramiko.Transport(('connect.nmb2.seetacloud.com', 32942)); t.connect(); t.auth_password('root', pw)
c = paramiko.SSHClient(); c._transport = t
cmd = 'which hf || true; which modelscope || true; /root/miniconda3/bin/python -m pip show huggingface_hub modelscope 2>/dev/null || true; df -h /root'
_i, o, e = c.exec_command(cmd, timeout=120)
text = o.read().decode(errors='replace') + e.read().decode(errors='replace')
Path('benchmark-results/gpu/tools.log').write_text('$ ' + cmd + '\n' + text, encoding='utf-8')
print(text)
