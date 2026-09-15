from pathlib import Path
import paramiko
pw=Path('.codex/gpu_password.txt').read_text().strip(); t=paramiko.Transport(('connect.nmb2.seetacloud.com',40423)); t.connect(); t.auth_password('root',pw); c=paramiko.SSHClient(); c._transport=t
cmd='/root/miniconda3/bin/python3.12 -m pip install transformers pyyaml huggingface_hub'
_i,o,e=c.exec_command(cmd,timeout=600); text=o.read().decode(errors='replace')+e.read().decode(errors='replace'); Path('benchmark-results/gpu-new/install.log').write_text('$ '+cmd+'\n'+text); print(text)
