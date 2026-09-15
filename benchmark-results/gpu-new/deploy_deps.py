from pathlib import Path
import paramiko
root=Path('.').resolve(); pw=(root/'.codex/gpu_password.txt').read_text().strip()
t=paramiko.Transport(('connect.nmb2.seetacloud.com',40423)); t.connect(); t.auth_password('root',pw); s=paramiko.SFTPClient.from_transport(t)
local=root/'benchmark-results/deps_bundle.zip'; remote='/root/bench/deps_bundle.zip'; s.put(str(local),remote); s.close()
c=paramiko.SSHClient(); c._transport=t
cmd="mkdir -p /root/bench/pydeps && cd /root/bench/pydeps && unzip -oq /root/bench/deps_bundle.zip -d /root/miniconda3/lib/python3.12/site-packages"
_i,o,e=c.exec_command(cmd,timeout=180); print((o.read()+e.read()).decode(errors='replace')); c.close(); t.close()
