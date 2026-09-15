from pathlib import Path
import paramiko
pw = Path('.codex/gpu_password.txt').read_text(encoding='utf-8').strip()
t = paramiko.Transport(('connect.nmb2.seetacloud.com', 32942)); t.connect(); t.auth_password('root', pw)
c = paramiko.SSHClient(); c._transport = t
cmd = 'ls -la /; ls -la /root/autodl-fs; ls -la /root/autodl-pub; du -sh /root/.cache /root/autodl-fs /root/autodl-pub 2>/dev/null'
_i, o, e = c.exec_command(cmd, timeout=180)
text = o.read().decode(errors='replace') + e.read().decode(errors='replace')
Path('benchmark-results/gpu/storage.log').write_text('$ ' + cmd + '\n' + text, encoding='utf-8')
print(text)
