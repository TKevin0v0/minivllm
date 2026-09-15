from pathlib import Path
import paramiko
pw=Path('.codex/gpu_password.txt').read_text().strip(); t=paramiko.Transport(('connect.nmb2.seetacloud.com',40423)); t.connect(); t.auth_password('root',pw); c=paramiko.SSHClient(); c._transport=t
cmd="find /root -type f -path '*/bin/python*' 2>/dev/null | head -50; find /opt /usr/local -type f -path '*/bin/python*' 2>/dev/null | head -50"
_i,o,e=c.exec_command(cmd,timeout=180); text=o.read().decode(errors='replace')+e.read().decode(errors='replace'); Path('benchmark-results/gpu-new/runtime.log').write_text('$ '+cmd+'\n'+text); print(text)
