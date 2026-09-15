from pathlib import Path
import paramiko
pw = Path('.codex/gpu_password.txt').read_text(encoding='utf-8').strip()
t = paramiko.Transport(('connect.nmb2.seetacloud.com', 32942)); t.connect(); t.auth_password('root', pw)
c = paramiko.SSHClient(); c._transport = t
cmd = "find /root /autodl-fs /autodl-pub /data /workspace -type f \\( -name '*.safetensors' -o -name 'config.json' \\) 2>/dev/null | head -200"
_i, o, e = c.exec_command(cmd, timeout=300)
text = o.read().decode(errors='replace') + e.read().decode(errors='replace')
Path('benchmark-results/gpu/weights.log').write_text('$ ' + cmd + '\n' + text, encoding='utf-8')
print(text)
